# 物化 IR 输出审计与复现

本次修复的是旧回归测试的决策协议和导出可读性，没有更改计算变换算法，
也没有通过复制另一个算子的函数、类型或硬件属性来改变输出。

## 截图对应什么

`python/test/unit/l_lite/test_materialization_numeric.py`读取
`test/Triton/hbv-loop-integer-carry-materialize.mlir`等手写MLIR测试。
这里的`integer_carry`不是Python前端生成的add_kernel。旧测试调用局部pass链，
输出materialized.ttir，再用triton.compile从后续后端阶段继续；这不能证明
Python源码完整入口，后者有独立的test_python_entry.py测试。

旧integer fixture带adapter_version=8的归档决策。当前triton-opt直接执行其
RUN命令会报`ordinary L subject set is not closed`。当前数值测试已改为保留
fixture主体、移除该归档module协议、从新鲜事实构建当前决策。三个fixture已
去掉过期协议，RUN/FileCheck仅检查源IR合法和入口函数；真正的变换结构断言
及数值检查在test_materialization_numeric.py中，不把源IR检查冒充变换检查。

## 属性不是变换

主L的`hbv/l_core/compiler_src/include/LCore/OrdinaryPasses.inc`中CoreFactsPass
将分析写入`tt.hbv.l.static_facts`。共享Python的`core/ttir.py`读取它构建事实，
主L的live_schedule_snapshot也使用它。因此这不是可以从正式链路随便删除的
纯日志。决策闭合、物化和验证分别由对应pass负责；事实字段不是优化证据。

新增共享`core/ir_export.py`保留原始字符串逐字不变，另导出
`*.reading.mlir`和`*.attributes.json`。阅读副本将内部字符串属性替换为报告
索引，明确标记**不是编译输入、不是完整原始产物**。原始值及解码JSON都在报告中。
原生硬件属性和IR操作不改。阅读副本能被方言解析器读取，不代表可用它重跑
依赖那些属性的L pass；真正继续编译必须使用原件。

## 已验证的实际结构

- integer：四次循环展开成四组load/mul，通过tt.join及reshape/trans构成归约
  输入，tt.reduce后加回非零初值7，再tt.store。未仅依赖成功标签。
- float：同样验证`7+3*sum`的结果，不能因为浮点与整数都是向量化就免测。
- nested：Bridge仍是外层两次迭代；内层factor2展开形成step2的循环，每组两个
  load相邻，之后进行运算。不是把Bridge循环误当成同级优化的内层对象，也
  不是声称factor2把trip4的整个循环全部消除了。

三个数值测试均校验输出、输入未被修改、输出两侧哨兵。最近一次3 passed
（4.39秒是测试总墙钟时间，不是kernel性能数据）。没有做性能比较。

同时补了操作级断言：integer没有残留for、有四次load、有归约结果加回；
float存在高秩mulf且没有错误复制的轴提示；nested有外内两循环且两load相邻。
这些是本次fixture的预期结构，不是对所有算子的准入规则。

`test_python_entry.py`中python_bridge_candidate与hinted_python_loop两个测试组
另跑10项全部通过（含GPU正确性），覆盖Python源码、Bridge factor2/4与
三种路线。它们验证Python候选编译入口；本次未重测整个算子库或所有默认入口。

## 本机复现

以下路径是本次环境；换机应替换为自己的项目及依赖目录。当前准备接口只验证
SM89，本次设备是RTX4090，不应复制参考截图的cuda:90。

```bash
cd /workspace/hbv-l-lite-public
CPATH=/usr/local/cuda/include CUDA_VISIBLE_DEVICES=1 L_LITE_RUN_NUMERIC=1 \
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc \
python -I -S -c 'import sys;sys.path[:0]=["/workspace/hbv-l-lite-public/python","/home/jinyiw/venv/lib/python3.10/site-packages"];import pytest;raise SystemExit(pytest.main(["python/test/unit/l_lite/test_materialization_numeric.py","-q","--basetemp=/tmp/l-lite-ir-audit-review"]))'
```

选择一个全新的basetemp目录，pytest会清理其指定的临时目录。`-I -S`避免共享
环境的.pth将另一个项目的Triton加载进来；测试也断言当前包和扩展属于本树。
CPATH是本地直接构建所需的CUDA头文件路径，最初未设置时报cuda.h缺失。

最新实际产物在`/tmp/l-lite-ir-audit-numeric-v4/test_public_materialization_nu0/`
（integer）、`nu1/`（float）、`nu2/`（nested），其中nu1/nu2表示同名前缀末尾
替换0为1/2。每目录有source.ttir、materialized.ttir、kernel.ttir、kernel.ttgir、
kernel.llir、kernel.ptx、kernel.cubin及阅读副本和属性报告。

```bash
build/l-lite-public/bin/triton-opt \
  /tmp/l-lite-ir-audit-numeric-v4/test_public_materialization_nu0/materialized.ttir -o /dev/null
build/l-lite-public/bin/triton-opt \
  /tmp/l-lite-ir-audit-numeric-v4/test_public_materialization_nu0/kernel.ttgir -o /dev/null
```

本次三个目录的完整TTIR/TTGIR及其阅读副本共12份均通过上述工具解析验证。
完整后端产物来自compile_selected→triton.compile，未额外伪造阶段或target属性。

## 关键片段与阶段界限

integer物化TTIR中的实际片段（省略位置标记）：

```mlir
%34 = tt.join %9, %17 : tensor<16xi32> -> tensor<16x2xi32>
%35 = tt.join %25, %33 : tensor<16xi32> -> tensor<16x2xi32>
%36 = tt.join %34, %35 : tensor<16x2xi32> -> tensor<16x2x2xi32>
// 经形状变换得到 %38，归约四个贡献，得到 %39。
%40 = arith.addi %cst_0, %39 : tensor<16xi32>
tt.store %3, %40 : tensor<16x!tt.ptr<i32>>
```

完整文件包含实际tt.reduce及其region。SSA名称是一次导出的编号，不是稳定接口。
materialized.ttir是在本项目物化/验证之后、TTGIR降低之前；kernel.ttgir来自
正常后端，含blocked/shared等layout和target属性。差异由实际生成链路确认，
不是仅看模块头部猜阶段。参考截图没有完整文件和生成链，不能确认其精确阶段。

## 验收范围和缺失参考

本次没有同算子、手工实现同一优化策略的Python参考文件，因而没有宣称与
手工优化版本结构等价；只有原fixture语义公式的数值核对。若要进一步做手工
优化结构等价对比，需提供同一integer_carry算法的手工展开/打包实现，并绑定
同输入、编译器、SM89与阶段；不能使用另一个f32 add_kernel替代。

当前临时修复范围已通过上述验证。未发现要求重写Python Kernel或额外插入
TTIR→TTGIR转换的证据。不能从13个测试推断全部物化覆盖或性能等价。
