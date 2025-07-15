# 金融术语标准化工具

一个基于LangChain + OpenAI + Milvus的金融术语标准化工具，支持CSV数据导入、向量化存储和语义相似性查询，配备现代化的Web界面。

## ✨ 功能特性

### 🚀 核心功能
- **CSV数据导入** - 支持批量导入金融术语数据
- **智能向量化** - 基于OpenAI text-embedding-3-large模型生成高质量嵌入向量
- **语义相似性查询** - 支持模糊查询，返回最相关的标准金融术语
- **可视化进度** - 实时显示嵌入生成和数据库写入进度

### 🎨 Web界面
- **现代化UI** - 基于Gradio的响应式Web界面
- **双Tab设计** - 术语查询和数据导入分离
- **实时进度条** - 分别显示OpenAI嵌入和Milvus写入进度
- **文件上传** - 支持拖拽上传CSV文件

### 🏗️ 架构设计
- **前后端分离** - 界面逻辑与数据处理完全分离
- **单一职责** - 每个模块职责明确，易于维护和扩展
- **依赖精简** - 只保留必要依赖，安装和部署更轻量

### 🔧 技术特性
- **自动配置** - 智能搜索并加载配置文件
- **性能优化** - 大批处理、智能进度显示和时间估算
- **测试模式** - 支持小批量数据验证配置
- **Milvus Lite** - 默认使用本地文件数据库，无需启动服务
- **完整测试** - 82%代码覆盖率，结构化测试套件

## 📦 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd finance_standard

# 安装依赖（精简版，只包含必要包）
pip install -r requirements.txt
```

### 依赖说明
```
# 核心依赖
gradio>=4.0.0          # Web界面
pandas>=2.0.0          # 数据处理
langchain-openai>=0.1.0  # OpenAI嵌入
pymilvus>=2.4.0        # Milvus数据库
python-dotenv>=1.0.0   # 环境变量
tqdm>=4.60.0           # 进度条

# 测试依赖
pytest>=7.0.0          # 测试框架

# 可选依赖
milvus-lite>=2.4.0     # 本地Milvus数据库
```

## ⚙️ 配置设置

### 方法一：配置向导（推荐）
```bash
# 运行配置向导，按提示设置
python setup_config.py
```

### 方法二：手动配置
创建 `config.env` 文件：
```bash
# 必需配置
OPENAI_API_KEY=your-openai-api-key-here

# 可选配置（使用默认值）
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=finance_term
BATCH_SIZE=500
INSERT_BATCH_SIZE=1000
```

## 🚀 使用方法

### Web界面（推荐）
```bash
# 启动Web界面
python src/finance_term_gradio_app.py

# 在浏览器中访问显示的URL（通常是 http://localhost:7860）
```

**Web界面功能：**
- **标准化术语查询** - 输入任意词语，返回最相近的5个标准金融术语
- **标准化术语导入** - 上传CSV文件，批量导入并向量化存储

### 命令行使用
```bash
# 直接导入数据（使用默认CSV文件）
python src/finance_term_loader.py

# 运行测试验证配置
python tests/run_tests.py
```

### 编程使用
```python
from src.finance_term_loader import FinanceTermLoader
import pandas as pd

# 初始化加载器
loader = FinanceTermLoader()

# 方式1：导入CSV文件
result = loader.load_finance_terms("data/万条金融标准术语.csv")
print(f"导入完成，共处理{result['total_terms']}条术语")

# 方式2：导入DataFrame（支持上传文件场景）
df = pd.read_csv("your_file.csv", header=None, names=['term', 'category'])
result = loader.process_and_import(df)

# 查询相似术语
results = loader.search_similar_terms("股票", top_k=5)
for item in results:
    print(f"{item['term']} - {item['category']} - {item['score']:.4f}")
```

## 📁 数据格式

### CSV文件格式
上传的CSV文件应无表头，包含两列：
```csv
股票,FINTERM
债券,FINTERM
基金,FINTERM
期货,FINTERM
```

### Milvus存储结构
- `id`: 自动生成的主键
- `term`: 金融术语名称（VARCHAR，最大500字符）
- `category`: 术语类别（VARCHAR，最大100字符）
- `embedding`: 嵌入向量（3072维FLOAT_VECTOR）

## 🎯 性能优化

### 处理速度
- **批处理优化** - 默认批大小500，减少API调用次数
- **并行处理** - 支持多线程处理，提升效率
- **进度可视化** - 实时显示处理进度和剩余时间估算

### 时间估算
- **小批量测试（100条）** - 约1-2分钟
- **标准数据集（15885条）** - 约1-2小时
- **自定义数据** - 根据数据量线性估算

### 优化建议
- 首次使用建议运行测试验证配置
- 增加 `BATCH_SIZE` 可提升速度（最大2048）
- 网络不稳定时减少 `BATCH_SIZE` 避免超时

## 🧪 测试

### 快速测试
```bash
# 运行所有快速测试（无需API密钥）
python tests/run_tests.py

# 运行特定测试
python tests/run_tests.py --test config    # 配置测试
python tests/run_tests.py --test loader    # 核心功能测试
python tests/run_tests.py --test batch     # 批处理测试
```

### 测试覆盖
- **82%代码覆盖率** - 超过行业平均水平
- **33个测试用例** - 覆盖配置、核心功能、批处理
- **Mock测试** - 不依赖外部服务的快速测试
- **集成测试** - 端到端功能验证

## 🏗️ 项目结构

```
finance_standard/
├── src/                          # 源代码
│   ├── finance_term_loader.py   # 核心数据处理模块
│   └── finance_term_gradio_app.py # Web界面模块
├── data/                         # 数据文件
│   └── 万条金融标准术语.csv
├── tests/                        # 测试套件
│   ├── test_configuration.py    # 配置测试
│   ├── test_finance_term_loader.py # 核心功能测试
│   ├── test_batch_processing.py # 批处理测试
│   └── run_tests.py             # 测试运行器
├── db/                          # 数据库文件（自动生成）
├── requirements.txt             # 依赖配置
├── setup_config.py             # 配置向导
├── config.env                  # 环境配置
└── README.md                   # 项目文档
```

## 🔧 环境变量配置

| 变量名 | 默认值 | 说明 | 必需 |
|--------|--------|------|------|
| `OPENAI_API_KEY` | 无 | OpenAI API密钥 | ✅ |
| `MILVUS_HOST` | `localhost` | Milvus服务器地址 | ❌ |
| `MILVUS_PORT` | `19530` | Milvus服务器端口 | ❌ |
| `MILVUS_COLLECTION_NAME` | `finance_term` | 集合名称 | ❌ |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | 嵌入模型 | ❌ |
| `BATCH_SIZE` | `500` | 嵌入批次大小 | ❌ |
| `INSERT_BATCH_SIZE` | `1000` | 插入批次大小 | ❌ |
| `MILVUS_USE_LITE` | `true` | 使用Milvus Lite | ❌ |

## 🏆 架构亮点

### 前后端分离
- **界面层** (`finance_term_gradio_app.py`) - 仅负责用户交互和进度展示
- **业务层** (`finance_term_loader.py`) - 处理所有数据操作和Milvus交互
- **清晰边界** - 职责分离，易于维护和测试

### 单一职责原则
- **配置管理** - 统一的环境变量处理
- **数据处理** - 专门的CSV读取和DataFrame处理
- **向量化** - 独立的嵌入生成逻辑
- **数据库操作** - 专门的Milvus连接和数据管理

### 现代化特性
- **响应式UI** - 基于Gradio的现代Web界面
- **实时反馈** - 进度条和状态更新
- **错误处理** - 完善的异常处理和用户提示
- **可扩展性** - 易于添加新功能和数据源

## 📝 开发指南

### 添加新功能
1. **后端逻辑** - 在 `FinanceTermLoader` 类中添加方法
2. **前端界面** - 在 `finance_term_gradio_app.py` 中添加UI组件
3. **测试覆盖** - 在 `tests/` 目录下添加对应测试

### 代码规范
- 遵循PEP 8代码风格
- 使用类型提示
- 编写docstring文档
- 保持测试覆盖率

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 提供统一的嵌入模型接口
- [Milvus](https://github.com/milvus-io/milvus) - 高性能向量数据库
- [Gradio](https://github.com/gradio-app/gradio) - 简单易用的Web界面框架
- [OpenAI](https://openai.com/) - 强大的文本嵌入模型

## 📞 支持

如有问题或建议，请：
- 创建Issue
- 发送邮件
- 查看项目Wiki 