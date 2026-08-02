# 工具与 Skill 环境问题记录

这份记录不是论文内容，也不是要求每次开工都重新检查一遍。它只保存已经实际碰到过的问题，后续遇到相同现象时先照这里处理，避免重复试错。

## 一、先记住三个总原则

1. **命令存在不等于命令可用。** `Get-Command` 只能说明系统找得到程序，真正能不能工作还要看一次最小运行。
2. **不要把一批外部程序塞进同一个长 PowerShell 命令。** 某一个程序卡住后，前面已经完成的结果也可能拿不到。MATLAB、MiKTeX、LibreOffice 这类程序应分别测试。
3. **普通说明不用脚本。** 只有处理大数据、批量文件和需要复现的计算时才写脚本。读文档、解释方案和查看少量状态时，直接用简单命令或自然语言完成。

## 二、PDF 与 MiKTeX

### 遇到的现象

`pdftotext`、`pdftoppm` 和 `pdfimages` 都能在 PATH 中找到，但在沙箱内运行时会提示：

```text
It seems that this is a fresh TeX installation.
Please finish the setup before proceeding.
```

同时还会报告无法创建：

```text
C:\Users\34788\AppData\Roaming\MiKTeX\2.9
```

### 真实原因

MiKTeX 本身已经完整安装。`initexmf --report` 显示当前为 MiKTeX 26.2，配置、路径和数据库都正常。报错是因为沙箱不能写用户配置目录，MiKTeX 因此给出了容易误解的“全新安装”提示。

### 后续做法

- 读取普通 PDF 时优先使用已经可用的 PyMuPDF 或 pdfplumber，不必调用 MiKTeX。
- 确实需要 `pdftotext`、`pdftoppm`、XeLaTeX 或 latexmk 时，直接申请沙箱外运行。
- 不要再次重装 MiKTeX，也不要把这个问题描述成“pdftotext 被其他进程占用”。

已经实际验证过的链路是：XeLaTeX 生成 PDF，随后 `pdftotext` 能正确读回正文。

## 三、PowerShell 下的 Node 命令

### 遇到的现象

直接运行 `npm`、`npx` 或 `codex` 时，PowerShell 可能优先找到同名 `.ps1` 文件，并因执行策略受限而报错。

### 后续做法

在 Windows PowerShell 中直接使用：

```powershell
npm.cmd
npx.cmd
codex.cmd
```

不要为了这件事反复修改 PowerShell 执行策略。当前 Node 依赖已经全局安装，用户级 `NODE_PATH` 指向：

```text
C:\Users\34788\AppData\Roaming\npm\node_modules
```

DOCX、PPTX、React 和 Sharp 模块均已通过实际加载与内存生成测试。

## 四、LibreOffice 与 Office Skill

### 遇到的现象

LibreOffice 安装后可以正常转换 DOCX，但 Office Skill 自带的包装器在 Windows 上访问 `socket.AF_UNIX`，导致 XLSX 重算报错：

```text
module 'socket' has no attribute 'AF_UNIX'
```

### 真实原因

包装器把 Linux 下的 `AF_UNIX` 和 `LD_PRELOAD` 检查直接用到了 Windows。当前 Anaconda Python 3.13 的 Windows `socket` 没有 `AF_UNIX`。

### 已做修复

在 DOCX、PPTX 和 XLSX Skill 的 `office/soffice.py` 中，Windows 会跳过 Linux shim：

```python
if os.name == "nt" or not hasattr(socket, "AF_UNIX"):
    return False
```

数学建模 Skill 内的 DOCX 工具原本已经有等价判断。若以后更新或重装这些 Skill，要先确认这项修复有没有被覆盖。

已经实际验证：

- DOCX 可以转换为 PDF；
- XLSX 中的公式可以由 LibreOffice 重算；
- 测试公式 `SUM(2, 3)` 的缓存值为 5，错误数为 0。

## 五、GTK、Cairo 与 SVG 渲染

### 遇到的现象

只安装 CairoSVG 的 Python 包仍会缺少 `libcairo-2.dll`。把 GTK 的 `bin` 目录放到 PATH 最前面，又会让其中的 Python 3.7 DLL 干扰 Anaconda Python 3.13。

### 后续做法

- GTK 运行库安装在 `C:\Program Files\Gtk-Runtime`。
- `C:\Program Files\Gtk-Runtime\bin` 只能追加在用户 PATH 末尾，不能放在最前面。
- 不再用 Conda 安装 Cairo；此前 Conda 求解长时间无结果，已经终止。

CairoSVG 和 svglib 都已实际把 SVG 渲染成 PNG。

## 六、Tesseract 中文识别

### 遇到的现象

Tesseract 安装后只有 `eng` 和 `osd`，而且普通用户不能把中文语言包写进 `Program Files`。

### 已做修复

中文语言数据放在用户目录：

```text
C:\Users\34788\AppData\Local\Tesseract-OCR\tessdata
```

用户环境变量 `TESSDATA_PREFIX` 已指向该目录。当前包含：

- `chi_sim`
- `eng`
- `osd`

已经用临时图片实际识别出“数学建模”。

## 七、OpenAI Docs 与论文搜索

### OpenAI Docs

直接获取 Codex 手册时，官方网站对 HEAD 请求返回过 HTTP 403。按照 Skill 自带说明，已经安装全局 MCP：

```text
openaiDeveloperDocs -> https://developers.openai.com/mcp
```

新会话启动后应优先使用该 MCP，不要反复重试被拒绝的手册脚本。

### 论文搜索

OpenAlex 已经通过真实查询。AnySearch 目前没有配置 `ANYSEARCH_API_KEY`，所以双引擎搜索暂时只能使用 OpenAlex 一侧。后续正式需要双引擎时再补密钥，不要在没有密钥时反复调用。

## 八、MATLAB 当前能力

MATLAB R2025a 可以在沙箱外正常启动。当前安装的产品只有：

- MATLAB
- Optimization Toolbox
- Symbolic Math Toolbox

Statistics and Machine Learning Toolbox 尚未安装。因此，`factoran`、`pca`、`lasso` 等函数不能直接假定可用。C 题第一版代码应先检查这些函数；若确实需要，再决定安装工具箱，不能写完代码以后才发现函数不存在。

## 九、检查到什么程度为止

工具第一次修好后做一次最小功能测试即可。后续没有出现新报错时，不需要每轮重新检查 31 份 Skill、219 个脚本或所有依赖。

正常工作时只检查当前任务会用到的工具。例如：

- 读 CSV 和跑 MATLAB，只看 MATLAB 与当前数据入口；
- 写 Word 时再看 DOCX 和 LibreOffice；
- 查文献时再看 OpenAlex、AnySearch；
- 做中文扫描 PDF 时再看 Tesseract。

这样既能保留已经修好的环境，也不会让工具检查反过来占据建模时间。

## 十、不要凭预估名称读取阶段产物

本轮整理阶段 1 说明时，第一次把实际文件 `stage1_validation_groups.csv` 预估成了 `stage1_internal_validation_groups.csv`，因此出现一次“路径不存在”。处理办法很简单：引用阶段结果前先看一眼该目录的真实文件名，再读取内容；文档中只写实际已经生成的路径，不凭记忆补全名称。

## 十一、MATLAB 工具箱命令行安装

### 2026-08-02：R2025a 输入模板第一次下载失败

使用官方 `matlab-install-products` 流程安装 Statistics and Machine Learning Toolbox。`mpm.exe` 已成功下载到 `C:\Users\34788\Downloads\mpm.exe`，但第一次从 MathWorks 下载 R2025a 官方输入模板时，`Invoke-WebRequest` 返回 `Access Denied`，参考号为 `18.4dbd2817.1785648852.681710ac`。按照安装流程只允许再重试一次；若第二次仍失败，则停止命令行安装并改由用户在 MATLAB App 中安装。

第二次下载仍返回 `Access Denied`，参考号为 `18.4dbd2817.1785648932.681bdcdb`。命令行安装已经停止，没有执行 `mpm install`，Statistics and Machine Learning Toolbox 仍未安装。临时工作目录 `C:\Users\34788\AppData\Local\Temp\mpm-20260802-132949` 保留了三份安装脚本，已经下载的 `mpm.exe` 也予以保留；下一步由用户在 MATLAB 的 App 或附加功能界面搜索并安装 **Statistics and Machine Learning Toolbox**。

### 2026-08-02：用户已通过 MATLAB 界面完成安装

`C:\software\professional\Matlab\appdata\prodcontents.json` 已列出 Statistics and Machine Learning Toolbox 25.1。随后在 MATLAB R2025a 中实际运行 `pca`、`factoran`、`lasso` 和 `zscore`，四个函数均成功，退出码为 0。后续 C 题可以正常使用统计与机器学习工具箱；此前的 CDN 报错只影响 mpm 命令行安装，不影响用户通过 MATLAB 界面安装后的使用。

## 十二、本地数模讲义目录名有一个空格

AGENTS 中写的是 `1.视频配套所有课件和代码（包含正课和更新的模型）`，磁盘上的真实名称是 `1. 视频配套所有课件和代码（包含正课和更新的模型）`，数字和“视频”之间多了一个空格。第一次按 AGENTS 原文访问时报“路径不存在”；后续直接使用带空格的真实路径，不把它误判成工作区外不可访问。

## 十三、MATLAB 跨行函数调用需要续行符

`run_stage2_baselines.m` 第一次运行时在汇总表的 `summary = table(` 处报“无效表达式”，退出码为 1。原因是函数参数从下一行继续书写时，上一行末尾没有写 `...`。已经改为 `summary = table( ...`；以后跨行写 MATLAB 函数调用时明确保留续行符，不按其他语言的括号换行习惯书写。

同一脚本第二次运行已完成基线计算，但在选择输出表列时退出。`validation(:, {"cell_line_id", ...})` 生成的是字符串元胞数组，不是 MATLAB 表允许的列索引类型；已经改为字符串数组 `validation(:, ["cell_line_id", ...])`。以后表列选择统一使用字符串数组或字符向量元胞数组，不混用两者。

第一次修补上述问题时，同一文件的补丁片段没有按原文件行号顺序排列，`apply_patch` 因而找不到前面的原文，文件未被部分修改。重新按从前到后的顺序排列后再应用；同一文件包含多处修改时，补丁片段也应保持原文顺序。

## 十四、Python 输出 PDF 数学符号时要指定 UTF-8

用 pdfplumber 读取本地 PCA 讲义时，PDF 已正常打开，但控制台按 GBK 输出数学符号 `−` 时触发 `UnicodeEncodeError`，导致同批读取中断。处理办法是在该次 Python 命令前设置 `PYTHONIOENCODING=utf-8` 后重读；这属于控制台编码问题，不是 PDF 内容损坏，也不需要改写原讲义。

## 十五、12 因子最大似然模型出现零唯一方差

第一次运行 `run_stage3_factor_candidates.m` 时，MATLAB 在 12 因子模型中警告“有些唯一方差为零，无法计算显著性”。此时 `factoran` 返回的 `stats` 结构不包含脚本预期的 `chisq` 字段，脚本随后退出，退出码为 1。后续不能只补一个字段判断就当作模型正常；要先检查唯一方差和实际返回字段，把零唯一方差视为候选因子数可能不合适的证据，再决定是否保留该候选。

## 十六、MATLAB 命令必须放入 shell 工具参数

第一次准备运行 `run_stage3_final_modules` 时，把命令文本直接放在工具调用外层，工具解析器返回 `SyntaxError: Unexpected string`。错误发生在 MATLAB 启动之前，没有生成或改动结果。后续统一通过 `shell_command` 的 `command` 参数传入 `matlab -batch` 命令。

## 十七、GitHub Issue/PR 操作缺少可用入口

启用新的 Git 工作流程后，C 题第 4 阶段应先创建真实 GitHub Issue，再建立实验分支。当前系统找不到 `gh` 命令，Codex 本轮也没有提供 GitHub Issue/PR 连接器，因此不能按新规则创建远程 Issue。

用户已明确允许安装 GitHub Skill、连接器或官方工具。随后尝试通过官方 `skill-installer` 查询 curated Skills，并尝试使用 `winget install --id GitHub.cli --exact` 安装 GitHub 官方 CLI；两次提权审批都在命令真正执行前因审批服务连接中断而失败，错误为 `stream disconnected before completion`。复查后 `gh` 仍未安装。

这个问题不影响已有仓库和结果，但会阻断正式模型步骤的 Issue、分支、PR 流程。当前没有创建第 4 阶段 Issue、分支或代码。后续可在审批服务恢复后重试官方 CLI 安装；安装成功后运行 `gh auth login`，只通过官方浏览器授权，不在聊天中传递访问令牌。

## 十八、PowerShell 会优先命中被执行策略拦截的 codex.ps1

直接运行 `codex mcp ...` 时，PowerShell 优先调用 `C:\Users\34788\AppData\Roaming\npm\codex.ps1`，系统执行策略禁止加载该脚本。改用同目录下的 `codex.cmd mcp ...` 可以正常运行，不需要修改整台电脑的 PowerShell 执行策略。当前 `codex.cmd mcp list` 显示命令行侧没有已配置的 MCP 服务器。

## 十九、GitHub CLI 已安装但曾读取到无效的旧登录状态

用户改用 GitHub CLI 后，已确认 `C:\Program Files\GitHub CLI\gh.exe` 存在；当前环境没有 `GITHUB_TOKEN` 或 `GH_TOKEN`，Codex 配置中也没有 GitHub MCP 段，旧的 MCP 令牌路径已移除。

Codex 工具会话第一次运行 `gh auth status` 时报告默认 Token 无效，但用户随后在自己的 PowerShell 中确认账号 `leo-grayrat` 已通过系统 keyring 登录，Token 权限包含 `repo` 和 `workflow`，并能成功读取目标仓库。此前结果应视为工具会话读取到的旧凭据状态，不能作为用户本机当前状态的结论。

## 二十、GitHub CLI 在用户终端可用，但 Codex shell 出站网络受限

用户在本机终端执行 `gh repo view leo-grayrat/math-modeling-xjtu-2026` 成功，说明 GitHub CLI 和权限配置本身正常。Codex shell 使用同一 `gh.exe` 执行只读仓库查询时，访问 `api.github.com` 报 `connectex ... socket ... forbidden`；随后提权重试又因 `moyuu.cc` 自动审批服务断开而未执行。

因此当前边界是：用户可以在本机 CLI 中完成 GitHub 操作，但 Codex 工具会话暂时不能直接代为访问 GitHub。这个问题与 Token 有效性无关，也没有创建 Issue、分支或 Pull Request。

## 二十二、Codex 侧 GitHub 访问仍有两层阻碍

用户本机的 `gh repo view` 已经成功，说明账号和仓库权限正常。Codex shell 中运行 `gh repo view` 时曾报 `connectex ... socket ... forbidden`，这是出站网络策略阻止访问 `api.github.com`。随后运行 `git ls-remote origin HEAD` 又报 `SEC_E_NO_CREDENTIALS`，因为 Codex 侧 Git 仍使用 Windows Credential Manager，尚未通过 `gh auth setup-git` 接入 CLI 的 keyring。

因此 `gh auth setup-git` 只能解决凭据衔接，不能解决 Codex 沙箱网络限制。当前不能从 Codex 工具会话直接访问 GitHub；需要平台恢复提权审批、放开该执行环境的出站网络，或由用户在自己的终端执行 GitHub 操作。

## 二十一、GitHub 多账号导致 Credential Manager 反复弹出选择框

用户观察到每次发送消息时都会出现 GitHub 的 `Select an account` 窗口，列表中同时有 `leo-grayrat` 和另一个账号。当前仓库远程地址使用 HTTPS，Git 全局凭据助手为 `manager`，没有配置 `credential.https://github.com.username`；`cmdkey` 中也没有可直接区分的 GitHub 条目。

这与 Codex 自动审批模型的 `stream disconnected` 错误不是同一个问题。前者是本机 Git Credential Manager 无法确定使用哪个 GitHub 账号，后者是模型服务审批请求断开。建议后续执行 `gh auth setup-git`，让 Git 使用 GitHub CLI 已登录的 keyring 凭据；如果仍弹窗，再在 Windows 凭据管理器中删除不需要的旧账号或为本仓库固定用户名。当前只完成诊断，没有修改凭据。

## 二十三、阶段5脚本首次运行时出现 MATLAB 运算符语法错误

准备阶段5“细胞个体修正”脚本后，第一次运行 `matlab -batch "addpath('C题建模/matlab'); run_stage5_cell_correction(42)"` 在脚本第236行报“运算符的使用无效”，尚未读取数据或生成阶段5结果。当前脚本已用 Git stash 暂存，阶段5没有继续推进；后续恢复时先定位该行的矩阵或字符串表达式，再重新运行，不把这次失败当成模型结论。

后续已经恢复脚本并修正语法，阶段 5 全量运行成功，退出码为 0。该问题只影响第一次启动，没有造成结果污染。

## 二十四、MATLAB 自动读取 CSV 时没有识别表头

阶段 5 第一次读取 `stage3_module_definitions.csv` 时，`readtable` 把列名识别成 `Var1` 至 `Var15`，脚本因此找不到 `top_pathways`，在建立基因功能归属前退出。文件本身的第一行确实是正确表头，不是结果文件损坏。

后续在这份文件的读取语句中明确指定：

```matlab
Delimiter=",", ReadVariableNames=true, Encoding="UTF-8"
```

再次运行后成功读取中文模块名称和英文通路字段，训练 607,578 行、验证 265,065 行，退出码为 0。以后读取含中文内容且必须保留固定列名的 CSV 时，先显式声明分隔符、表头和 UTF-8 编码，不依赖 MATLAB 自动判断。

## 二十五、PowerShell 吃掉 MATLAB 字符串数组中的双引号

第 6 步准备检查 MATLAB 数据处理和统计工具箱时，第一次把 `check_matlab_env(["data","statistics"])` 放进 PowerShell 的双引号命令。PowerShell 传参后，MATLAB 实际收到的内容在 `([` 处中断，报“该语句不完整”，退出码为 1；检查函数尚未运行，因此不能把这次报错理解成缺少工具箱。

后续改用不含嵌套双引号的写法：

```matlab
check_matlab_env(string({'data','statistics'}))
```

在 PowerShell 调用 MATLAB 时，遇到字符串数组优先使用 `string` 包装字符向量元胞，避免在多层引号中继续试错。

改用上述写法后，环境检查成功：MATLAB R2025a、MATLAB 基础功能和 Statistics and Machine Learning Toolbox 均可用，退出码为 0。

## 二十六、PowerShell 管道把 GitHub Issue 中文正文变成问号

创建阶段 6 Issue #9 时，标题作为普通命令参数传递，因此中文正常；正文则通过 PowerShell 的字符串管道送给 `gh issue create --body-file -`。GitHub 上的正文几乎全部显示为 `?`，CLI 重新读取后也确认服务器收到的就是问号，说明损坏发生在本机管道传输阶段，不是网页字体或浏览器显示问题。

后续不再用 PowerShell 字符串管道传递含中文的 Issue、评论或 PR 正文。统一先生成 UTF-8 Markdown 临时文件，再使用 `gh ... --body-file <文件>`；上传后立即读回一次正文，确认中文没有损坏，再删除临时文件。Issue #9 的正文按此方法修复，并另建 Sub-issue 保存用户报告和处理结果。

## 二十七、论文双引擎搜索中 OpenAlex 返回 429

为 Issue #16 查找双边回归和冷启动方法时，按“双引擎论文搜索” Skill 同时调用 OpenAlex 与 AnySearch。OpenAlex 返回 `HTTP 429: Too Many Requests`，AnySearch 仍正常返回 DOI 候选。本轮只用候选条目确认方法有正式依据，没有把单引擎结果写成已交叉验证文献。正式论文引用前再核对作者、年份、期刊和全文内容，不为本次计划变更反复重试接口。

同时发现 AGENTS.md 中本地讲义目录写为 `1.视频配套...`，实际目录名在 `1.` 后多一个空格，即 `1. 视频配套...`。已按实际目录成功读取岭回归与 LASSO 讲义，不需要安装或移动资料。

## 二十八、P1 后续任务一度被转换成无法读取的密文

阶段 7 第二次修正后，向原 P1 质检者发送复验范围时，后续任务正文在对方会话中显示为一整段 `gAAAA...` 字符串，质检者无法读取并明确拒绝猜测。随后同一质检者收到了可执行的复验任务，并按当前工作区完成了 36 特征底座、标签隔离、真实 smoke 和 4 项单元测试检查，最终返回 PASS；没有因此跳过 P1 或改用作者自检冒充独立验收。

若再次出现同类情况，应先确认质检者是否实际收到明文任务，不能连续发送不可读内容，也不能根据加密字符串推测任务范围。
