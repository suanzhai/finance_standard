#!/usr/bin/env python3
"""
金融术语加载工具配置设置脚本

这个脚本帮助你快速设置配置文件，特别是OpenAI API密钥
"""

import os
import sys
from pathlib import Path

def setup_config():
    """设置配置文件"""
    print("🔧 金融术语加载工具 - 配置设置向导")
    print("=" * 60)
    
    # 获取项目根目录
    project_root = Path(__file__).parent
    config_file = project_root / "config.env"
    
    # 检查是否已存在配置文件
    if config_file.exists():
        print(f"✅ 发现现有配置文件: {config_file}")
        
        # 检查是否有有效的API密钥
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'OPENAI_API_KEY=your-openai-api-key-here' in content:
                    print("⚠️  发现默认API密钥，需要更新")
                    need_update = True
                elif 'OPENAI_API_KEY=' in content and 'sk-' in content:
                    print("✅ 发现已配置的API密钥")
                    need_update = False
                else:
                    print("⚠️  API密钥配置可能不完整")
                    need_update = True
        except Exception as e:
            print(f"❌ 读取配置文件失败: {e}")
            need_update = True
            
        if not need_update:
            choice = input("\n是否要重新配置API密钥？(y/N): ").strip().lower()
            if choice not in ['y', 'yes']:
                print("配置设置已取消")
                return
    else:
        print("📝 未找到配置文件，将创建新的配置文件")
        need_update = True
    
    # 获取API密钥
    print("\n🔑 配置OpenAI API密钥")
    print("请访问 https://platform.openai.com/api-keys 获取你的API密钥")
    
    while True:
        api_key = input("\n请输入你的OpenAI API密钥: ").strip()
        
        if not api_key:
            print("❌ API密钥不能为空")
            continue
        
        if not api_key.startswith('sk-'):
            print("❌ OpenAI API密钥应该以 'sk-' 开头")
            continue
        
        if len(api_key) < 40:
            print("❌ API密钥长度似乎不够")
            continue
        
        break
    
    # 获取其他可选配置
    print("\n⚙️  其他配置（可选，直接回车使用默认值）")
    
    milvus_host = input("Milvus服务器地址 [localhost]: ").strip() or "localhost"
    milvus_port = input("Milvus服务器端口 [19530]: ").strip() or "19530"
    collection_name = input("Milvus集合名称 [finance_term]: ").strip() or "finance_term"
    embedding_model = input("OpenAI嵌入模型 [text-embedding-3-large]: ").strip() or "text-embedding-3-large"
    batch_size = input("批处理大小 [100]: ").strip() or "100"
    
    # 生成配置文件内容
    config_content = f"""# 金融术语加载工具环境变量配置文件
# 自动生成于 {os.path.basename(__file__)}

# =============================================================================
# 必需配置
# =============================================================================

# OpenAI API密钥
OPENAI_API_KEY={api_key}

# =============================================================================
# Milvus数据库配置
# =============================================================================

# Milvus服务器地址
MILVUS_HOST={milvus_host}

# Milvus服务器端口
MILVUS_PORT={milvus_port}

# Milvus集合名称
MILVUS_COLLECTION_NAME={collection_name}

# =============================================================================
# 数据处理配置
# =============================================================================

# 数据库文件路径（用于Milvus Lite）
DB_FILE=db/milvus_lite.db

# 嵌入模型配置
EMBEDDING_MODEL={embedding_model}

# 嵌入向量维度（text-embedding-3-large对应3072）
EMBEDDING_DIM=3072

# 性能配置
BATCH_SIZE={batch_size}
INSERT_BATCH_SIZE=1000
TEST_MODE_LIMIT=

# =============================================================================
# LangChain高级配置（可选）
# =============================================================================

# LangChain追踪设置（用于调试和监控）
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=your-langsmith-api-key

# 其他嵌入模型提供商API密钥（按需设置）
# COHERE_API_KEY=your-cohere-api-key
# ANTHROPIC_API_KEY=your-anthropic-api-key
"""
    
    # 写入配置文件
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"\n✅ 配置文件已保存到: {config_file}")
        print("\n🚀 配置完成！现在可以运行以下命令:")
        print("   python src/example_usage.py        # 运行主程序")
        print("   python src/search_example.py       # 搜索示例")
        print("   python src/multi_provider_example.py  # 多提供商演示")
        
    except Exception as e:
        print(f"❌ 保存配置文件失败: {e}")
        return False
    
    return True

def validate_config():
    """验证配置文件"""
    project_root = Path(__file__).parent
    config_file = project_root / "config.env"
    
    if not config_file.exists():
        print("❌ 未找到配置文件")
        return False
    
    print("🔍 验证配置文件...")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查API密钥
        if 'OPENAI_API_KEY=your-openai-api-key-here' in content:
            print("❌ API密钥仍然是默认值，请重新配置")
            return False
        elif 'OPENAI_API_KEY=sk-' in content:
            print("✅ API密钥配置正确")
        else:
            print("⚠️  API密钥配置可能有问题")
            return False
        
        print("✅ 配置文件验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 验证配置文件失败: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        validate_config()
        return
    
    if setup_config():
        print("\n" + "=" * 60)
        print("🎉 设置完成！请确保Milvus服务正在运行，然后就可以使用工具了。")
        
        # 提供快速测试选项
        test_choice = input("\n是否要进行快速测试验证配置？(y/N): ").strip().lower()
        if test_choice in ['y', 'yes']:
            print("\n🧪 进行配置测试...")
            try:
                from dotenv import load_dotenv
                from langchain_openai import OpenAIEmbeddings
                
                # 加载配置
                config_file = Path(__file__).parent / "config.env"
                load_dotenv(config_file)
                
                # 测试嵌入模型
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                test_result = embeddings.embed_query("测试")
                
                print(f"✅ 配置测试通过！嵌入向量维度: {len(test_result)}")
                
            except Exception as e:
                print(f"❌ 配置测试失败: {e}")
                print("请检查API密钥是否正确以及网络连接")

if __name__ == "__main__":
    main() 