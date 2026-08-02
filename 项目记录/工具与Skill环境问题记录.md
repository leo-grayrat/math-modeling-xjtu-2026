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
