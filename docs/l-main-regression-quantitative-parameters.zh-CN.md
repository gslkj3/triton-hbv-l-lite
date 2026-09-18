# 主 L 定量关系：冻结参数附录

由 `render_regression_quantitative_appendix.py` 从已有文件生成，未重新训练。

所有数字均为自然对数坐标上的参数，不是硬件常数。z为标准化坐标，h为分段项；定义见主报告。

输入维数：17；展开项：170；非零权重：165。

## 1. 标准化与节点

| 输入字段 | 均值 μ | 尺度 σ | 节点 k |
|---|---:|---:|---:|
| `load_lower` | 6.864095964519108 | 3.0757061651664257 | 6.932447891572509 |
| `load_upper` | 6.9396506783595777 | 2.9771200785686713 | 6.932447891572509 |
| `store_lower` | 5.430972764531564 | 3.686763938256465 | 5.5490760848952201 |
| `store_upper` | 5.430972764531564 | 3.686763938256465 | 5.5490760848952201 |
| `arith_lower` | 8.8880133630239495 | 3.1191987980445592 | 8.4362000322067061 |
| `arith_upper` | 8.9501591743818576 | 3.0831033202848581 | 8.4924905787758664 |
| `special_lower` | 4.5167324109913141 | 4.2102678439627281 | 4.2046926193909657 |
| `special_upper` | 4.7177740411269138 | 4.2540446618323999 | 4.8675344504555822 |
| `reduce_lower` | 1.43638522610875 | 2.2013947338244839 | 0.69314718055994529 |
| `reduce_upper` | 1.43638522610875 | 2.2013947338244839 | 0.69314718055994529 |
| `other_lower` | 8.7144054104818078 | 2.9240891411318337 | 8.3795390261174418 |
| `other_upper` | 8.7866165172401143 | 2.8863526019982246 | 8.4364168813889489 |
| `programs` | 4.76427631381822 | 2.5068926673995402 | 4.8598124043616719 |
| `static_address_without_backedge_sites` | 0.17791884265420208 | 0.53019968059720413 | 0 |
| `static_address_through_backedge_sites` | 0.15610510165447167 | 0.50147967769217816 | 0 |
| `log1p_input_bytes_per_logical_program` | 6.5900406178236794 | 2.0492744250936106 | 6.2402758451707694 |
| `log1p_output_bytes_per_logical_program` | 5.368056409681877 | 2.8963675970344842 | 6.2402758451707694 |

## 2. 全部权重（顺序与冻结推理器一致）

| 序号 | 基函数 | 权重 |
|---:|---|---:|
| 0 | `z[load_lower]` | -0.17067351554926591 |
| 1 | `z[load_upper]` | -0.23173369802947139 |
| 2 | `z[store_lower]` | -0.054755115212882642 |
| 3 | `z[store_upper]` | -0.054755115212883433 |
| 4 | `z[arith_lower]` | 0.28625428531304153 |
| 5 | `z[arith_upper]` | 0.25428804343915745 |
| 6 | `z[special_lower]` | -0.11860352781572656 |
| 7 | `z[special_upper]` | -0.16173648347016306 |
| 8 | `z[reduce_lower]` | 0.17263787921296447 |
| 9 | `z[reduce_upper]` | 0.17263787921295595 |
| 10 | `z[other_lower]` | 0.058958777787222062 |
| 11 | `z[other_upper]` | 0.046341642923889849 |
| 12 | `z[programs]` | 0.096453263087413352 |
| 13 | `z[static_address_without_backedge_sites]` | -0.037311416510518718 |
| 14 | `z[static_address_through_backedge_sites]` | 0.012886253233875211 |
| 15 | `z[log1p_input_bytes_per_logical_program]` | 0 |
| 16 | `z[log1p_output_bytes_per_logical_program]` | 0 |
| 17 | `h[load_lower]` | -0.13703288774471903 |
| 18 | `h[load_upper]` | -0.1913674354536013 |
| 19 | `h[store_lower]` | 0.041193834709673846 |
| 20 | `h[store_upper]` | 0.041193834709670508 |
| 21 | `h[arith_lower]` | 0.23422288145402029 |
| 22 | `h[arith_upper]` | -0.050949701420665287 |
| 23 | `h[special_lower]` | -0.35530663187788797 |
| 24 | `h[special_upper]` | -0.018314533578798354 |
| 25 | `h[reduce_lower]` | 0.17263787921298854 |
| 26 | `h[reduce_upper]` | 0.17263787921298829 |
| 27 | `h[other_lower]` | 0.78011515696637401 |
| 28 | `h[other_upper]` | 0.067670275239446159 |
| 29 | `h[programs]` | 0.24943475304184862 |
| 30 | `h[static_address_without_backedge_sites]` | -0.03731141651051078 |
| 31 | `h[static_address_through_backedge_sites]` | 0.012886253233907474 |
| 32 | `h[log1p_input_bytes_per_logical_program]` | 0 |
| 33 | `h[log1p_output_bytes_per_logical_program]` | 0 |
| 34 | `z[load_lower] × z[load_upper]` | -0.017681425061396899 |
| 35 | `z[load_lower] × z[store_lower]` | 0.027250493658898471 |
| 36 | `z[load_lower] × z[store_upper]` | 0.027250493658880135 |
| 37 | `z[load_lower] × z[arith_lower]` | 0.01316126876880718 |
| 38 | `z[load_lower] × z[arith_upper]` | 0.033320098167991752 |
| 39 | `z[load_lower] × z[special_lower]` | -0.051307523918922175 |
| 40 | `z[load_lower] × z[special_upper]` | -0.0036720132718133531 |
| 41 | `z[load_lower] × z[reduce_lower]` | -0.041939242095243676 |
| 42 | `z[load_lower] × z[reduce_upper]` | -0.041939242095243981 |
| 43 | `z[load_lower] × z[other_lower]` | 0.011651476016870997 |
| 44 | `z[load_lower] × z[other_upper]` | 0.026307360333759779 |
| 45 | `z[load_lower] × z[programs]` | 0.18685030892432367 |
| 46 | `z[load_lower] × z[static_address_without_backedge_sites]` | -0.059989934647040541 |
| 47 | `z[load_lower] × z[static_address_through_backedge_sites]` | -0.032514581602477263 |
| 48 | `z[load_lower] × z[log1p_input_bytes_per_logical_program]` | 0.085741764067333359 |
| 49 | `z[load_lower] × z[log1p_output_bytes_per_logical_program]` | 0.17763593737459119 |
| 50 | `z[load_upper] × z[store_lower]` | -0.018490438624606143 |
| 51 | `z[load_upper] × z[store_upper]` | -0.018490438624606188 |
| 52 | `z[load_upper] × z[arith_lower]` | -0.0073272148029136395 |
| 53 | `z[load_upper] × z[arith_upper]` | 0.005998233625946028 |
| 54 | `z[load_upper] × z[special_lower]` | -0.018435282768182473 |
| 55 | `z[load_upper] × z[special_upper]` | 0.029545169660592206 |
| 56 | `z[load_upper] × z[reduce_lower]` | -0.030954473479544952 |
| 57 | `z[load_upper] × z[reduce_upper]` | -0.030954473479544987 |
| 58 | `z[load_upper] × z[other_lower]` | -0.014639488691407463 |
| 59 | `z[load_upper] × z[other_upper]` | -0.0053352231578431752 |
| 60 | `z[load_upper] × z[programs]` | 0.25126333173275311 |
| 61 | `z[load_upper] × z[static_address_without_backedge_sites]` | -0.052396016915435425 |
| 62 | `z[load_upper] × z[static_address_through_backedge_sites]` | -0.026951370987320977 |
| 63 | `z[load_upper] × z[log1p_input_bytes_per_logical_program]` | 0.071097649847619168 |
| 64 | `z[load_upper] × z[log1p_output_bytes_per_logical_program]` | 0.12468347345558199 |
| 65 | `z[store_lower] × z[store_upper]` | -0.17648155987802247 |
| 66 | `z[store_lower] × z[arith_lower]` | -0.023152852636225834 |
| 67 | `z[store_lower] × z[arith_upper]` | -0.0097864390052442591 |
| 68 | `z[store_lower] × z[special_lower]` | -0.040620690783060964 |
| 69 | `z[store_lower] × z[special_upper]` | 0.0048086272242024095 |
| 70 | `z[store_lower] × z[reduce_lower]` | -0.13866973980387648 |
| 71 | `z[store_lower] × z[reduce_upper]` | -0.13866973980387642 |
| 72 | `z[store_lower] × z[other_lower]` | -0.010867522575256057 |
| 73 | `z[store_lower] × z[other_upper]` | 0.0038127157695396845 |
| 74 | `z[store_lower] × z[programs]` | 0.0022333201054916115 |
| 75 | `z[store_lower] × z[static_address_without_backedge_sites]` | -0.023130154819044842 |
| 76 | `z[store_lower] × z[static_address_through_backedge_sites]` | 0.026540619238186361 |
| 77 | `z[store_lower] × z[log1p_input_bytes_per_logical_program]` | 0.019517926831788407 |
| 78 | `z[store_lower] × z[log1p_output_bytes_per_logical_program]` | -0.20121329430781928 |
| 79 | `z[store_upper] × z[arith_lower]` | -0.023152852636228745 |
| 80 | `z[store_upper] × z[arith_upper]` | -0.0097864390052448732 |
| 81 | `z[store_upper] × z[special_lower]` | -0.040620690783060867 |
| 82 | `z[store_upper] × z[special_upper]` | 0.0048086272242024025 |
| 83 | `z[store_upper] × z[reduce_lower]` | -0.13866973980387648 |
| 84 | `z[store_upper] × z[reduce_upper]` | -0.1386697398038812 |
| 85 | `z[store_upper] × z[other_lower]` | -0.010867522575256208 |
| 86 | `z[store_upper] × z[other_upper]` | 0.0038127157695396416 |
| 87 | `z[store_upper] × z[programs]` | 0.0022333201054908543 |
| 88 | `z[store_upper] × z[static_address_without_backedge_sites]` | -0.023130154819035211 |
| 89 | `z[store_upper] × z[static_address_through_backedge_sites]` | 0.0265406192382057 |
| 90 | `z[store_upper] × z[log1p_input_bytes_per_logical_program]` | 0.019517926831783963 |
| 91 | `z[store_upper] × z[log1p_output_bytes_per_logical_program]` | -0.20121329430781446 |
| 92 | `z[arith_lower] × z[arith_upper]` | 0.026725522484567013 |
| 93 | `z[arith_lower] × z[special_lower]` | 0.038020109842337092 |
| 94 | `z[arith_lower] × z[special_upper]` | 0.082637863508272694 |
| 95 | `z[arith_lower] × z[reduce_lower]` | 0.019573345693778884 |
| 96 | `z[arith_lower] × z[reduce_upper]` | 0.019573345693778912 |
| 97 | `z[arith_lower] × z[other_lower]` | 0.0073594998614074179 |
| 98 | `z[arith_lower] × z[other_upper]` | 0.014830004243464142 |
| 99 | `z[arith_lower] × z[programs]` | -0.016121183351669167 |
| 100 | `z[arith_lower] × z[static_address_without_backedge_sites]` | 0.0069491788529308209 |
| 101 | `z[arith_lower] × z[static_address_through_backedge_sites]` | 0.00056485449143338048 |
| 102 | `z[arith_lower] × z[log1p_input_bytes_per_logical_program]` | -0.05740836331105096 |
| 103 | `z[arith_lower] × z[log1p_output_bytes_per_logical_program]` | -0.12890410184164802 |
| 104 | `z[arith_upper] × z[special_lower]` | 0.054802708063223037 |
| 105 | `z[arith_upper] × z[special_upper]` | 0.09907351384861933 |
| 106 | `z[arith_upper] × z[reduce_lower]` | 0.034616842711339788 |
| 107 | `z[arith_upper] × z[reduce_upper]` | 0.034616842711339781 |
| 108 | `z[arith_upper] × z[other_lower]` | 0.019251295721116606 |
| 109 | `z[arith_upper] × z[other_upper]` | 0.02607744390768095 |
| 110 | `z[arith_upper] × z[programs]` | -0.037576082949981056 |
| 111 | `z[arith_upper] × z[static_address_without_backedge_sites]` | 0.0085494308777007803 |
| 112 | `z[arith_upper] × z[static_address_through_backedge_sites]` | 0.0012216803383354271 |
| 113 | `z[arith_upper] × z[log1p_input_bytes_per_logical_program]` | -0.032028341187338223 |
| 114 | `z[arith_upper] × z[log1p_output_bytes_per_logical_program]` | -0.11563359924707503 |
| 115 | `z[special_lower] × z[special_upper]` | 0.3240149753703836 |
| 116 | `z[special_lower] × z[reduce_lower]` | 0.076901737157070604 |
| 117 | `z[special_lower] × z[reduce_upper]` | 0.076901737157070493 |
| 118 | `z[special_lower] × z[other_lower]` | -0.046605361632143863 |
| 119 | `z[special_lower] × z[other_upper]` | -0.033120236694752662 |
| 120 | `z[special_lower] × z[programs]` | 0.10450110706831907 |
| 121 | `z[special_lower] × z[static_address_without_backedge_sites]` | 0.013074167047166917 |
| 122 | `z[special_lower] × z[static_address_through_backedge_sites]` | -0.0083724781004645907 |
| 123 | `z[special_lower] × z[log1p_input_bytes_per_logical_program]` | -0.12223471337985002 |
| 124 | `z[special_lower] × z[log1p_output_bytes_per_logical_program]` | -0.01504539280849445 |
| 125 | `z[special_upper] × z[reduce_lower]` | 0.11352657371767708 |
| 126 | `z[special_upper] × z[reduce_upper]` | 0.11352657371767703 |
| 127 | `z[special_upper] × z[other_lower]` | -0.0043779753659071137 |
| 128 | `z[special_upper] × z[other_upper]` | 0.0089504475241305552 |
| 129 | `z[special_upper] × z[programs]` | 0.063169550694372328 |
| 130 | `z[special_upper] × z[static_address_without_backedge_sites]` | 0.014821872010881403 |
| 131 | `z[special_upper] × z[static_address_through_backedge_sites]` | -0.0070686814719876879 |
| 132 | `z[special_upper] × z[log1p_input_bytes_per_logical_program]` | -0.030671941434205024 |
| 133 | `z[special_upper] × z[log1p_output_bytes_per_logical_program]` | 0.020814142984314613 |
| 134 | `z[reduce_lower] × z[reduce_upper]` | -0.093222442649555856 |
| 135 | `z[reduce_lower] × z[other_lower]` | -0.0552664348543178 |
| 136 | `z[reduce_lower] × z[other_upper]` | -0.036376828671257605 |
| 137 | `z[reduce_lower] × z[programs]` | -0.2021315926171838 |
| 138 | `z[reduce_lower] × z[static_address_without_backedge_sites]` | -0.12938171989288832 |
| 139 | `z[reduce_lower] × z[static_address_through_backedge_sites]` | -0.052788040204299842 |
| 140 | `z[reduce_lower] × z[log1p_input_bytes_per_logical_program]` | 0.10655396609870518 |
| 141 | `z[reduce_lower] × z[log1p_output_bytes_per_logical_program]` | 0.031042335552168891 |
| 142 | `z[reduce_upper] × z[other_lower]` | -0.055266434854317828 |
| 143 | `z[reduce_upper] × z[other_upper]` | -0.036376828671257626 |
| 144 | `z[reduce_upper] × z[programs]` | -0.20213159261718397 |
| 145 | `z[reduce_upper] × z[static_address_without_backedge_sites]` | -0.12938171989288841 |
| 146 | `z[reduce_upper] × z[static_address_through_backedge_sites]` | -0.05278804020429987 |
| 147 | `z[reduce_upper] × z[log1p_input_bytes_per_logical_program]` | 0.10655396609870522 |
| 148 | `z[reduce_upper] × z[log1p_output_bytes_per_logical_program]` | 0.03104233555213836 |
| 149 | `z[other_lower] × z[other_upper]` | 0.021963874019031208 |
| 150 | `z[other_lower] × z[programs]` | -0.17106240920010948 |
| 151 | `z[other_lower] × z[static_address_without_backedge_sites]` | 0.010814452148722713 |
| 152 | `z[other_lower] × z[static_address_through_backedge_sites]` | 0.024241057703468533 |
| 153 | `z[other_lower] × z[log1p_input_bytes_per_logical_program]` | 0.093872792888980941 |
| 154 | `z[other_lower] × z[log1p_output_bytes_per_logical_program]` | 0.15916636919812893 |
| 155 | `z[other_upper] × z[programs]` | -0.20141306498600045 |
| 156 | `z[other_upper] × z[static_address_without_backedge_sites]` | 0.011943277978250991 |
| 157 | `z[other_upper] × z[static_address_through_backedge_sites]` | 0.024948909332482037 |
| 158 | `z[other_upper] × z[log1p_input_bytes_per_logical_program]` | 0.08703407305649935 |
| 159 | `z[other_upper] × z[log1p_output_bytes_per_logical_program]` | 0.15278409871270876 |
| 160 | `z[programs] × z[static_address_without_backedge_sites]` | -0.17545213566292514 |
| 161 | `z[programs] × z[static_address_through_backedge_sites]` | -0.10167687719354962 |
| 162 | `z[programs] × z[log1p_input_bytes_per_logical_program]` | 0.36112357704093567 |
| 163 | `z[programs] × z[log1p_output_bytes_per_logical_program]` | -0.099818739104367227 |
| 164 | `z[static_address_without_backedge_sites] × z[static_address_through_backedge_sites]` | -0.021804793984450125 |
| 165 | `z[static_address_without_backedge_sites] × z[log1p_input_bytes_per_logical_program]` | -0.070971435104344532 |
| 166 | `z[static_address_without_backedge_sites] × z[log1p_output_bytes_per_logical_program]` | 0.091769824346070103 |
| 167 | `z[static_address_through_backedge_sites] × z[log1p_input_bytes_per_logical_program]` | 0.11446469876699189 |
| 168 | `z[static_address_through_backedge_sites] × z[log1p_output_bytes_per_logical_program]` | -0.11647459171642698 |
| 169 | `z[log1p_input_bytes_per_logical_program] × z[log1p_output_bytes_per_logical_program]` | 0 |

## 3. 可复算的已保存源码评分

复核已有支持评分 492 条：全部与冻结公式一致（容差1e-10）；这只是算术一致性，不是精度或收益验证。

示例入口 18，kernel `_amp_foreach_non_finite_check_and_unscale_kernel`，候选 `l.loop-candidate.d5ed26552973dbfd5d254e92`。

对数加速比=-0.087287008193181762；预测加速比=0.91641403954763512。不是本次新测收益。

| 输入字段 | Original | Candidate |
|---|---:|---:|
| `load_lower` | 6.9334230257307148 | 7.6260827580723802 |
| `load_upper` | 6.9334230257307148 | 7.6260827580723802 |
| `store_lower` | 6.932447891572509 | 7.6251071482389001 |
| `store_upper` | 6.932447891572509 | 7.6251071482389001 |
| `arith_lower` | 9.329633378164166 | 9.8763218844633585 |
| `arith_upper` | 9.329633378164166 | 9.8763218844633585 |
| `special_lower` | 6.932447891572509 | 7.6251071482389001 |
| `special_upper` | 6.932447891572509 | 7.6251071482389001 |
| `reduce_lower` | 0 | 0 |
| `reduce_upper` | 0 | 0 |
| `other_lower` | 9.0114015093587909 | 9.5709475744425205 |
| `other_upper` | 9.0114015093587909 | 9.5709475744425205 |
| `programs` | 13.862944564872768 | 13.169798337985775 |
| `static_address_without_backedge_sites` | 0 | 0 |
| `static_address_through_backedge_sites` | 0 | 0 |
| `log1p_input_bytes_per_logical_program` | 7.6270574170189338 | 7.6270574170189338 |
| `log1p_output_bytes_per_logical_program` | 7.6251071482389001 | 7.6251071482389001 |

| 基函数 | 对本次对数加速比的贡献 |
|---|---:|
| `z[load_lower]` | 0.038436269672648925 |
| `z[load_upper]` | 0.053915393741460579 |
| `z[store_lower]` | 0.0102872432401989 |
| `z[store_upper]` | 0.010287243240199048 |
| `z[arith_lower]` | -0.050170552693735028 |
| `z[arith_upper]` | -0.045089747632801665 |
| `z[special_lower]` | 0.019512257761143053 |
| `z[special_upper]` | 0.026334531327657437 |
| `z[reduce_lower]` | 0 |
| `z[reduce_upper]` | 0 |
| `z[other_lower]` | -0.01128219781296289 |
| `z[other_upper]` | -0.0089837547670462252 |
| `z[programs]` | 0.026668958048902267 |
| `z[static_address_without_backedge_sites]` | -0 |
| `z[static_address_through_backedge_sites]` | 0 |
| `z[log1p_input_bytes_per_logical_program]` | 0 |
| `z[log1p_output_bytes_per_logical_program]` | 0 |
| `h[load_lower]` | 0.030860283216334689 |
| `h[load_upper]` | 0.044523738754914617 |
| `h[store_lower]` | -0.0077393864665864032 |
| `h[store_upper]` | -0.0077393864665857761 |
| `h[arith_lower]` | -0.041051233183170117 |
| `h[arith_upper]` | 0.0090342791896704435 |
| `h[special_lower]` | 0.058453864848071101 |
| `h[special_upper]` | 0.002982039969796142 |
| `h[reduce_lower]` | 0 |
| `h[reduce_upper]` | 0 |
| `h[other_lower]` | -0.14928080004556549 |
| `h[other_upper]` | -0.013118506798909919 |
| `h[programs]` | 0.068967754453083313 |
| `h[static_address_without_backedge_sites]` | -0 |
| `h[static_address_through_backedge_sites]` | 0 |
| `h[log1p_input_bytes_per_logical_program]` | 0 |
| `h[log1p_output_bytes_per_logical_program]` | 0 |
| `z[load_lower] × z[load_upper]` | 0.0010108330176901951 |
| `z[load_lower] × z[store_lower]` | -0.0037677079630615202 |
| `z[load_lower] × z[store_upper]` | -0.0037677079630589849 |
| `z[load_lower] × z[arith_lower]` | -0.00099111662930845387 |
| `z[load_lower] × z[arith_upper]` | -0.0023873113931537062 |
| `z[load_lower] × z[special_lower]` | 0.0087208649663495039 |
| `z[load_lower] × z[special_upper]` | 0.0005786374574283875 |
| `z[load_lower] × z[reduce_lower]` | -0.0061626667232761957 |
| `z[load_lower] × z[reduce_upper]` | -0.0061626667232762408 |
| `z[load_lower] × z[other_lower]` | -0.00081887998716389707 |
| `z[load_lower] × z[other_upper]` | -0.0017248663322344136 |
| `z[load_lower] × z[programs]` | -0.13992602731887735 |
| `z[load_lower] × z[static_address_without_backedge_sites]` | -0.0045335245249537477 |
| `z[load_lower] × z[static_address_through_backedge_sites]` | -0.0022793815350758339 |
| `z[load_lower] × z[log1p_input_bytes_per_logical_program]` | -0.0097713188212822533 |
| `z[load_lower] × z[log1p_output_bytes_per_logical_program]` | -0.031174074688968673 |
| `z[load_upper] × z[store_lower]` | 0.0025530203518950097 |
| `z[load_upper] × z[store_upper]` | 0.0025530203518950158 |
| `z[load_upper] × z[arith_lower]` | 0.00053746067860574291 |
| `z[load_upper] × z[arith_upper]` | -0.00041699917641897983 |
| `z[load_upper] × z[special_lower]` | 0.0031602835702290531 |
| `z[load_upper] × z[special_upper]` | -0.004687826700652634 |
| `z[load_upper] × z[reduce_lower]` | -0.00469915766483253 |
| `z[load_upper] × z[reduce_upper]` | -0.0046991576648325352 |
| `z[load_upper] × z[other_lower]` | 0.00099185758465804644 |
| `z[load_upper] × z[other_upper]` | 0.00033514411132666859 |
| `z[load_upper] × z[programs]` | -0.19615687274755847 |
| `z[load_upper] × z[static_address_without_backedge_sites]` | -0.0040907632501276815 |
| `z[load_upper] × z[static_address_through_backedge_sites]` | -0.0019519478457077056 |
| `z[load_upper] × z[log1p_input_bytes_per_logical_program]` | -0.0083707525908424292 |
| `z[load_upper] × z[log1p_output_bytes_per_logical_program]` | -0.022605812618011655 |
| `z[store_lower] × z[store_upper]` | 0.033236431653893533 |
| `z[store_lower] × z[arith_lower]` | 0.0030308755405726957 |
| `z[store_lower] × z[arith_upper]` | 0.0012590527060753419 |
| `z[store_lower] × z[special_lower]` | 0.0083560046392495563 |
| `z[store_lower] × z[special_upper]` | -0.00093629905824648102 |
| `z[store_lower] × z[reduce_lower]` | -0.016999220946492141 |
| `z[store_lower] × z[reduce_upper]` | -0.016999220946492134 |
| `z[store_lower] × z[other_lower]` | 0.0014450172380243357 |
| `z[store_lower] × z[other_upper]` | -0.00049567081319664157 |
| `z[store_lower] × z[programs]` | -0.0011553853937226184 |
| `z[store_lower] × z[static_address_without_backedge_sites]` | -0.0014582612462531199 |
| `z[store_lower] × z[static_address_through_backedge_sites]` | 0.0015522051552660612 |
| `z[store_lower] × z[log1p_input_bytes_per_logical_program]` | -0.0018556397582287951 |
| `z[store_lower] × z[log1p_output_bytes_per_logical_program]` | 0.02945904237266439 |
| `z[store_upper] × z[arith_lower]` | 0.0030308755405730765 |
| `z[store_upper] × z[arith_upper]` | 0.0012590527060754209 |
| `z[store_upper] × z[special_lower]` | 0.0083560046392495355 |
| `z[store_upper] × z[special_upper]` | -0.00093629905824647961 |
| `z[store_upper] × z[reduce_lower]` | -0.016999220946492141 |
| `z[store_upper] × z[reduce_upper]` | -0.01699922094649272 |
| `z[store_upper] × z[other_lower]` | 0.0014450172380243557 |
| `z[store_upper] × z[other_upper]` | -0.00049567081319663604 |
| `z[store_upper] × z[programs]` | -0.0011553853937222268 |
| `z[store_upper] × z[static_address_without_backedge_sites]` | -0.0014582612462525125 |
| `z[store_upper] × z[static_address_through_backedge_sites]` | 0.0015522051552671922 |
| `z[store_upper] × z[log1p_input_bytes_per_logical_program]` | -0.0018556397582283725 |
| `z[store_upper] × z[log1p_output_bytes_per_logical_program]` | 0.029459042372663682 |
| `z[arith_lower] × z[arith_upper]` | -0.002078031508467628 |
| `z[arith_lower] × z[special_lower]` | -0.005805228941182506 |
| `z[arith_lower] × z[special_upper]` | -0.011803517158339327 |
| `z[arith_lower] × z[reduce_lower]` | 0.0022383858988438833 |
| `z[arith_lower] × z[reduce_upper]` | 0.0022383858988438868 |
| `z[arith_lower] × z[other_lower]` | -0.00057722409328167718 |
| `z[arith_lower] × z[other_upper]` | -0.0011133349205884873 |
| `z[arith_lower] × z[programs]` | 0.0088426776068451889 |
| `z[arith_lower] × z[static_address_without_backedge_sites]` | 0.00040870773866980405 |
| `z[arith_lower] × z[static_address_through_backedge_sites]` | 3.0817487834385219e-05 |
| `z[arith_lower] × z[log1p_input_bytes_per_logical_program]` | 0.0050916404221250646 |
| `z[arith_lower] × z[log1p_output_bytes_per_logical_program]` | 0.017605617722072368 |
| `z[arith_upper] × z[special_lower]` | -0.0082839682543938819 |
| `z[arith_upper] × z[special_upper]` | -0.013991603663170867 |
| `z[arith_upper] × z[reduce_lower]` | 0.0040050904214410663 |
| `z[arith_upper] × z[reduce_upper]` | 0.0040050904214410654 |
| `z[arith_upper] × z[other_lower]` | -0.0014533495043837551 |
| `z[arith_upper] × z[other_upper]` | -0.001878735213801101 |
| `z[arith_upper] × z[programs]` | 0.021061693795607375 |
| `z[arith_upper] × z[static_address_without_backedge_sites]` | 0.00050871149689478922 |
| `z[arith_upper] × z[static_address_through_backedge_sites]` | 6.7433112703192349e-05 |
| `z[arith_upper] × z[log1p_input_bytes_per_logical_program]` | 0.0028739021832022678 |
| `z[arith_upper] × z[log1p_output_bytes_per_logical_program]` | 0.015978041694826667 |
| `z[special_lower] × z[special_upper]` | -0.066701159803074159 |
| `z[special_lower] × z[reduce_lower]` | 0.00825503794708571 |
| `z[special_lower] × z[reduce_upper]` | 0.0082550379470856979 |
| `z[special_lower] × z[other_lower]` | 0.0073629906745037409 |
| `z[special_lower] × z[other_upper]` | 0.0051646221566685295 |
| `z[special_lower] × z[programs]` | -0.041066217863504499 |
| `z[special_lower] × z[static_address_without_backedge_sites]` | 0.0007217826360638352 |
| `z[special_lower] × z[static_address_through_backedge_sites]` | -0.00042877326269854464 |
| `z[special_lower] × z[log1p_input_bytes_per_logical_program]` | 0.010176305762966749 |
| `z[special_lower] × z[log1p_output_bytes_per_logical_program]` | 0.0019288617020075025 |
| `z[special_upper] × z[reduce_lower]` | 0.012061133778828149 |
| `z[special_upper] × z[reduce_upper]` | 0.012061133778828144 |
| `z[special_upper] × z[other_lower]` | 0.00064494928753611365 |
| `z[special_upper] × z[other_upper]` | -0.0012993300988333185 |
| `z[special_upper] × z[programs]` | -0.025393967457755243 |
| `z[special_upper] × z[static_address_without_backedge_sites]` | 0.0008098473011473662 |
| `z[special_upper] × z[static_address_through_backedge_sites]` | -0.00035827769188932025 |
| `z[special_upper] × z[log1p_input_bytes_per_logical_program]` | 0.0025272286445306652 |
| `z[special_upper] × z[log1p_output_bytes_per_logical_program]` | -0.0026409718553383719 |
| `z[reduce_lower] × z[reduce_upper]` | -0 |
| `z[reduce_lower] × z[other_lower]` | -0.0069004862398324264 |
| `z[reduce_lower] × z[other_upper]` | -0.0046013397628859101 |
| `z[reduce_lower] × z[programs]` | 0.036466688531685214 |
| `z[reduce_lower] × z[static_address_without_backedge_sites]` | -0 |
| `z[reduce_lower] × z[static_address_through_backedge_sites]` | -0 |
| `z[reduce_lower] × z[log1p_input_bytes_per_logical_program]` | 0 |
| `z[reduce_lower] × z[log1p_output_bytes_per_logical_program]` | 0 |
| `z[reduce_upper] × z[other_lower]` | -0.0069004862398324299 |
| `z[reduce_upper] × z[other_upper]` | -0.0046013397628859127 |
| `z[reduce_upper] × z[programs]` | 0.036466688531685242 |
| `z[reduce_upper] × z[static_address_without_backedge_sites]` | -0 |
| `z[reduce_upper] × z[static_address_through_backedge_sites]` | -0 |
| `z[reduce_upper] × z[log1p_input_bytes_per_logical_program]` | 0 |
| `z[reduce_upper] × z[log1p_output_bytes_per_logical_program]` | 0 |
| `z[other_lower] × z[other_upper]` | -0.0015745697643501869 |
| `z[other_lower] × z[programs]` | 0.10495211740843678 |
| `z[other_lower] × z[static_address_without_backedge_sites]` | 0.00069443602807855228 |
| `z[other_lower] × z[static_address_through_backedge_sites]` | 0.0014439779507941554 |
| `z[other_lower] × z[log1p_input_bytes_per_logical_program]` | -0.0090901416085001786 |
| `z[other_lower] × z[log1p_output_bytes_per_logical_program]` | -0.023734724196786322 |
| `z[other_upper] × z[programs]` | 0.12658205928449939 |
| `z[other_upper] × z[static_address_without_backedge_sites]` | 0.00077694895242626683 |
| `z[other_upper] × z[static_address_through_backedge_sites]` | 0.0015055728859646321 |
| `z[other_upper] × z[log1p_input_bytes_per_logical_program]` | -0.0085381040253960545 |
| `z[other_upper] × z[log1p_output_bytes_per_logical_program]` | -0.02308087454073203 |
| `z[programs] × z[static_address_without_backedge_sites]` | 0.016279095297831676 |
| `z[programs] × z[static_address_through_backedge_sites]` | 0.0087513505601165935 |
| `z[programs] × z[log1p_input_bytes_per_logical_program]` | 0.050527829024405314 |
| `z[programs] × z[log1p_output_bytes_per_logical_program]` | -0.021507446192259552 |
| `z[static_address_without_backedge_sites] × z[static_address_through_backedge_sites]` | -0 |
| `z[static_address_without_backedge_sites] × z[log1p_input_bytes_per_logical_program]` | -0 |
| `z[static_address_without_backedge_sites] × z[log1p_output_bytes_per_logical_program]` | 0 |
| `z[static_address_through_backedge_sites] × z[log1p_input_bytes_per_logical_program]` | 0 |
| `z[static_address_through_backedge_sites] × z[log1p_output_bytes_per_logical_program]` | -0 |
| `z[log1p_input_bytes_per_logical_program] × z[log1p_output_bytes_per_logical_program]` | 0 |

## 4. 输入及实现摘要

| 文件 | SHA-256 |
|---|---|
| `carried-export-v1.json` | `2f952152d985e8dfce3e3622f9ee4a142988e45705f072b007562b07c8949644` |
| `source-global-v1.json` | `1afbd19fefc2a9debab01623b9e3ddf11c082b0db0dc854b450714f6e4f63a47` |
| `source_service_adapter.py` | `7f9212cb675d8713a2bd5ffe9dc9648f9893c3e5b07a7d9aab188bdcd6eaf3d3` |
| `frozen_paired_hinge.py` | `085dec3a720b6e70c0eb593e0a9e94141552ce0bdbd68993f52c667c46174515` |
| `paired_service_potential.py` | `4b6630babee3155b222f71d9ed188e344ee615ddc14b4002a268de01dbf7cece` |
| `hinge_service.py` | `00fcd6692432868b52cd8c372c4e99f42c284d8672db0a76e0c5f32a32e06663` |
| `carried_supply_service.py` | `9c5b953cddc6ac51630857a56e42928b303bab7e3ab8d784a413c3d223aab9ec` |
| `render_regression_quantitative_appendix.py` | `0d346968e1a2c909ecb0819a910d9981a252e9b6bea025a12c43b67c6f46a393` |
