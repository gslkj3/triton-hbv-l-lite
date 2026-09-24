# H20 执行交接：L-lite 完整候选编译与两库测试

## 分工和当前状态

本分支已同步非预测Python核心、规则式C++ Pass及必要原生信息传播修改。
H20架构接入由H20上的sol负责；本次没有放开SM90准入，没有H20性能结论。
当前前端仍明确限制SM89。不能只删除限制就声称移植完成。

本地独立公开分支已重建，12项GPU测试通过（4.68秒）：Python Bridge2/4、
两种program数量、带原生提示的tl.range三路线与两个BLOCK，以及真实原生
autotune的默认/显式constexpr绑定。编译和测试均不导入主L项目或预测模型。

**尚未完成：两个算子库所有典型shape的批量适配和收益报告。**现有入口是
真实Python kernel测试/调用入口，不能宣称已交付两库一键全量收益测试。

## 已有测试入口

文件：`python/test/unit/l_lite/test_python_entry.py`。
请先按仓库原生构建流程安装当前分支，不要借用旧wheel里的libtriton。
编译并发最多4。源码开发模式需要原生setup过程安装backend语言包链接，
例如`triton.language.extra.cuda`；CMake单独构建不会完成所有Python包装步骤。

在独立安装环境里，配置已安装的Triton包与CUDA工具绝对路径后执行：

```bash
CUDA_VISIBLE_DEVICES=0 L_CORE_TEST_GPU=1 python -I -S \
  l_lite_core/compiler_src/run_build_tests.py \
  --triton-package /absolute/path/to/triton \
  --dependency-site /absolute/path/to/site-packages \
  --ptxas /absolute/path/to/cuda/bin/ptxas \
  --libdevice /absolute/path/to/cuda/nvvm/libdevice/libdevice.10.bc \
  --cuda-include /absolute/path/to/cuda/include \
  -- python/test/unit/l_lite/test_python_entry.py -q
```

以上测试当前目标写为89，H20执行前须完成下节目标绑定与测试修改。隔离启动
入口避免共享Python的.pth自动注入另一份Triton；入口会检查实际extension。

## 用户kernel调用入口

无预测、无搜索的默认路径：`triton.l_lite.core.default_entry.default_kernel`。
参数为kernel、实参、grid、binding及可选kernel_kwargs/target；Bridge默认1，
只提交原生默认stage的软件流水候选，无法流水时仍继续普通原生编译。
它不会调用预测器或autotune，不是预测失败后的回退。默认入口并未全局替换
原生JITFunction.run；使用者需明确调用此入口。已有循环内的显式优化提示由
经过验证的流水计划接管；不支持流水的其他循环仍遵守原生编译行为。

`triton.l_lite.core.lite_entry.autotune_kernel`接受已装饰的`@triton.jit`函数和
实际参数；不用用户手写AST签名。`kernel_kwargs`传constexpr和原生编译选项，
`grid`可为元组或接受已绑定参数的函数。

```python
from triton.l_lite.core.lite_entry import autotune_kernel
from triton.l_lite.core.compiler import CompilerBinding
from triton.l_lite.core.state import Route

result = autotune_kernel(
    kernel, *args,
    kernel_kwargs=kernel_kwargs,
    grid=grid,
    bridge_factors=(1, 2, 4, 8),
    route_factors={Route.PIPELINE: (2, 3, 4),
                  Route.REORDER: (2, 4, 8),
                  Route.VECTORIZE: (2, 4, 8)},
    binding=CompilerBinding(
        'bound-by-entry',
        '7c56a5e40f7fd928dfd5c72902d5def0097db73a',
        'replace-with-current-build-identity'),
)
```

factor集合是可配置搜索范围，不是Pass白名单；候选合法性由统一规则决定。
源码身份由真实AST自动覆盖绑定，构建身份请填本机提交与extension哈希。
原生autotune不自动保证数值正确，测试适配器必须提供参考验证。原地操作需
传reset_to_zero/restore_value，并检查别名关系，不能重复测量中改变输入语义。

返回值：`population`为候选与拒绝，`timings`为原生测量结果，`selected`为选择，
`compilation_records`包含逐候选编译耗时/错误，`elapsed_ns`包含本次准备、搜索
与最终调用的墙钟耗时。它不是纯kernel时间；单位为纳秒。

## H20 sol 执行顺序

1. 固定当前提交、Triton基础版本、CUDA/驱动及GPU身份。检查frontend.py和
   ttir.py的目标绑定，按SM90原生选项处理，不套用SM89数值系数。L-lite无预测系数。
2. 将测试目标改为实际设备并完成编译/数值验证。流水是否真正物化需查看后端
   事实，不以请求stage或TTIR通过代替async机器指令证据。
3. 冷缓存再跑；验证range/tl.range、不同factor、动态及嵌套最内层循环。
   Bridge不得专用route；三选一对普通循环进行选择；不能跳过原生前后缀。
4. 固定FlagGems和FlagGems-vLLM提交，从库自身测试参数收集算子/shape/dtype。
   每个算子选择小/常用/大、对齐/尾部等适用代表点，公开所选清单和未覆盖项；
   不宣称有限清单覆盖所有可能shape。不要默认只用--quick的单一小shape。
5. 将真实库kernel调用接到上述入口；处理libentry缓存、外层autotune、warmup和
   graph捕获，禁止递归调优。库函数含多个kernel时保留逐kernel信息；算子级
   加速比必须测量整段库调用，不能取最快子kernel代替整个算子。
6. 每算子+shape单独进程，默认900秒超时，超时记账继续。启动前记录环境，
   收益测试可共享但需重复检查波动；不把无效或噪声结果列为确认收益。
7. 报告JSON/CSV/Markdown：库及版本、算子名、完整shape/stride/dtype、kernel、
   候选与赢家route/factor、Original/赢家耗时、加速比、搜索总耗时、状态原因。
   加速比=Original耗时/赢家耗时；Original选择为1倍。不混用旧L-lite实测。
8. 生成全部结果和正收益子表。大于1倍与达到1.0175物质门槛分别统计；失败、
   跳过、超时、未进入候选编译器保留完整分母。多kernel搜索成本按实际发生
   的搜索累加，冷/热缓存分别标注，不能把编译开销隐去。

## 不能改的边界

不引入主L预测模型；不按算子名/shape定制Pass；不提前剔除资源超限候选来
降低autotune开销；不把不支持或数值错误回退后的原生成功算成候选成功。
工程缺口按共性修复，单独记录；H20无测试证据的部分不能标已完成。
