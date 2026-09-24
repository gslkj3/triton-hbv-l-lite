# 物化 IR 输出审计与复现

当前任务不仅修复旧回归测试，还在迁移整个物化编译链的信息传递方式：
分析报告不再作为 JSON 字符串放进 IR；编译需要的决策采用紧凑的有类型属性。
迁移尚未全部验收，尤其最终模块属性的生命周期、独立构建和完整回归仍待收尾。
没有通过复制另一个算子的函数、类型或硬件属性来改变输出。

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

早期实现由 CoreFactsPass 把整份报告写入 `tt.hbv.l.static_facts`，再由
Python 读取。当前实现已停止这种传递方式：`core/ttir.py` 通过
`query_l_planning_facts` 直接取得结构化结果；旧 Facts Pass 调用会明确报错，
提示迁移到查询接口，不会悄悄继续写入报告。Bridge 的发现报告也通过查询返回。

`l_lite_core/compiler_src/include/LCore/TypedPlan.h` 只编码物化所需的决策字段。
完整审计合同在 Python 侧提交时检查，不整份存入 IR。验证 Pass 消费完闭合决策后
删除它。运行参数和 Bridge 网格除数也已改用有类型属性，不接受旧 JSON 字符串。
这些协议检查不能替代变换本身：决策闭合、物化和验证仍由对应 Pass 完成。

后端访存观察通过 `run_with_l_backend_observations` 在正常 PassManager 执行期间
收集，返回到 `metadata["l_backend_observations"]`；LowerLoops 不再把
`tt.hbv.l.backend_copy_width_facts` 写入 TTGIR。采集作用域隔离线程和嵌套编译，
异常退出会恢复前一个作用域。数值测试另存 `kernel.backend-observations.json`，
便于查看后端报告，而不占据模块头部。相关实现见原生 `Schedule.h`、
`LowerLoops.cpp` 和项目的 `PythonPassBindings.inc`。

辅助导出工具 `core/ir_export.py` 保留原始字符串逐字不变，另导出
`*.reading.mlir`和`*.attributes.json`。阅读副本将内部字符串属性替换为报告
索引，明确标记**不是编译输入、不是完整原始产物**。原始值及解码JSON都在报告中。
原生硬件属性和IR操作不改。阅读副本能被方言解析器读取，不代表可用它重跑
依赖那些属性的L pass；真正继续编译必须使用原件。
阅读副本只是辅助工具，不能作为“全链路不传 JSON”已经完成的证据。

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

以上结构和数值结果保留为早期验证记录，不代表新增全链路迁移已经全部完成。
2026-09-24，后端观察的隔离/异常测试与原生源码、运行时补丁安装测试合计
6 项通过；它们验证报告传递和安装，不是性能测试，也不证明所有算子覆盖。
未发现要求重写 Python Kernel 或额外插入 TTIR→TTGIR 转换的证据。
随后已将网格映射的快照读取改为支持链外显式上下文，并在验证后清理已消费的
模块配置。2026-09-24 新运行的 18 项 Python 入口和数值测试通过；三个数值目录
位于 `/tmp/l-lite-final-module-clean-v1/test_public_materialization_nu{0,1,2}/`。
其原始物化 TTIR 为 `module {`，原始 TTGIR 的模块头部如下（真实目标为 SM89）：

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
```

这不是阅读副本。函数和操作上仍保留后续编译及证据检查所需的信息。

### 各类信息存在哪里，何时结束使用

| 信息 | 当前传递方式 | 最后使用位置 |
|---|---|---|
| Bridge 运行整数、构造因子、网格除数 | 有类型的整数、字典和整数数组属性；不是 JSON 字符串 | 准备阶段提取及合法性检查；网格映射另存在调用者的 PreparedIR 中，验证后清除模块属性 |
| 可构造对象、循环清单、静态事实 | 只读查询返回结构化对象 | 调用者自行保留报告；不会写回 IR |
| 已选路线、循环成员、factor、必要守卫引用 | 紧凑的有类型决策属性 | 决策闭合、物化、验证；成功验证后删除闭合决策 |
| 完整审计合同 | Python 侧对象，提交时检查 | 不进入 IR；与紧凑决策并非两套独立决策规则 |
| 默认路线说明 | 编译结果 metadata | 供调用者诊断，不参与后端代码生成 |
| 后端访存宽度及流水观察 | 编译作用域内收集，返回 metadata，按需另存报告 | 后验验证和开发监督，不作为免费预编译输入 |
| 目标架构、warp/CTA 配置 | 原生编译器属性 | 后端需要，保留；由真实编译配置生成 |

项目仍使用 JSON 保存独立报告、历史数据或 Python 侧审计合同。这不等于通过 IR
传递 JSON：禁止的是把整份序列化报告嵌入程序属性、要求后续 Pass 再从字符串中
解析。保留的普通字符串用于路线名、来源标识等紧凑字段，不承载整份分析报告。

默认路径的路线说明也已从模块属性迁到编译结果的 `metadata.l_default_route`。
原生 JIT 测试检查默认/关闭模式的数值结果、缓存切换、预热，以及 TTIR/TTGIR
模块头不再携带项目诊断属性；与运行时补丁安装合计 16 项测试通过。
独立构建与剩余回归尚待完成，不宣称可交付结案。

随后启用所有本目录测试开关并绑定 fixture 目录，
`python/test/unit/l_lite` 整目录 **85 项通过、无跳过**（2026-09-24）。
测试包括三种路线的静态、动态和嵌套循环，两种块大小以及整数/浮点类型，
也包括默认调用、缓存切换和 autotune 接线。这是当前测试范围内的正确性证据，
不是性能等价或所有真实算子均受支持的证明。独立原生源码构建仍在进行。

扩大主项目回归后发现：原生 AxisInfo 查询会给被调用函数的参数补充属性。
为遵守只读接口约定，两个查询接口现在都在临时 IR 副本上分析，不将这些副作用
写回调用者。新增两项 helper 查询测试后，本目录再次完整运行 **87 项通过**；
主项目另有 **1137 项通过、125 项跳过**，不把跳过项目计为通过。该修复没有改
物化算法或预测参数；独立包仍须重新编译并验证。

## 独立包最终验证（2026-09-24）

本节更新前文中“独立构建仍待完成”的阶段状态：现已从固定原生 commit 重新
构建成功，再次构建确认无遗漏增量，并安装当前运行时补丁。新包位于
`/workspace/l-core-ir-clean-AyKjwH/python/triton`，没有使用旧扩展。

通过隔离运行器核对实际导入路径后，CPU 接口/物化相关 **41 项通过**，
GPU 完整 Python 编译入口 **51 项通过**。测试覆盖三路线、默认编译、缓存和
autotune 接线；测试文件来自项目，但被测扩展和共享运行时来自独立构建目录。
这完成了本次信息传递迁移的独立构建验证，不代表性能等价、模型收益目标完成，
也不代表测试集以外的所有程序均合法可优化。
