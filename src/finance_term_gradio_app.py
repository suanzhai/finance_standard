import gradio as gr
from finance_term_loader import FinanceTermLoader
import os
from dotenv import load_dotenv
from pathlib import Path

# 自动加载config.env（或.env）
load_dotenv(dotenv_path="config.env")

# 初始化加载器（假设环境变量已配置好）
loader = FinanceTermLoader()

# 查询功能保持不变
def search_terms(query):
    try:
        results = loader.search_similar_terms(query, top_k=5)
        # 转为表格展示
        if not results:
            return "未找到相关术语。"
        table = "| 术语 | 类别 | 相似度 |\n|---|---|---|\n"
        for item in results:
            table += f"| {item['term']} | {item['category']} | {item['score']:.4f} |\n"
        return table
    except Exception as e:
        return f"查询出错: {e}"

# 新的导入函数，带进度条和文件上传
def import_terms_with_progress(file):
    if file is None:
        yield "请先上传CSV文件。", 0, 0
        return
    import pandas as pd
    df = pd.read_csv(file, header=None, names=['term', 'category'])
    embed_progress = 0
    insert_progress = 0
    total_embed = len(df['term'])
    total_insert = len(df)
    # 进度回调函数
    def on_embed_progress(done, total):
        nonlocal embed_progress
        embed_progress = min(done / total, 1)
        yield_msg = f"OpenAI嵌入进度: {done}/{total}"
        # 只在嵌入阶段更新
        yield (yield_msg, embed_progress, 0)
    def on_insert_progress(done, total):
        nonlocal insert_progress
        insert_progress = min(done / total, 1)
        yield_msg = f"Milvus写入进度: {done}/{total}"
        # 只在插入阶段更新
        yield (yield_msg, 1, insert_progress)
    # 由于yield不能直接在回调里用，需用生成器代理
    # 嵌入进度
    embed_batches = []
    def embed_proxy(done, total):
        embed_batches.append((done, total))
    insert_batches = []
    def insert_proxy(done, total):
        insert_batches.append((done, total))
    # 调用loader
    import threading
    result = {}
    def run_loader():
        nonlocal result
        result = loader.process_and_import(df, on_embed_progress=embed_proxy, on_insert_progress=insert_proxy)
    t = threading.Thread(target=run_loader)
    t.start()
    last_embed = 0
    last_insert = 0
    while t.is_alive() or embed_batches or insert_batches:
        # 优先显示嵌入进度
        if embed_batches:
            done, total = embed_batches.pop(0)
            embed_progress = min(done / total, 1)
            yield (f"嵌入进度: {done}/{total}", embed_progress, 0)
            last_embed = embed_progress
        elif insert_batches:
            done, total = insert_batches.pop(0)
            insert_progress = min(done / total, 1)
            yield (f"Milvus插入进度: {done}/{total}", 1, insert_progress)
            last_insert = insert_progress
        else:
            import time
            time.sleep(0.1)
    # 完成
    yield (f"导入成功！\n\n- 术语总数: {result.get('total_terms', '-') }\n- 集合名: {result.get('collection_name', '-') }\n- 嵌入模型: {result.get('embedding_model', '-') }\n- 嵌入维度: {result.get('embedding_dim', '-') }", 1, 1)

with gr.Blocks() as demo:
    gr.Markdown("# 金融术语标准化工具")
    with gr.Tabs():
        with gr.TabItem("标准化术语查询"):
            gr.Markdown("输入一个词，返回最相近的5个标准金融术语：")
            with gr.Row():
                inp = gr.Textbox(label="输入词语", placeholder="请输入金融术语或词语")
            out = gr.Markdown()
            btn = gr.Button("搜索")
            btn.click(fn=search_terms, inputs=inp, outputs=out)
        with gr.TabItem("标准化术语导入"):
            gr.Markdown("上传CSV文件并导入Milvus数据库：")
            file_input = gr.File(label="上传CSV文件", file_types=[".csv"])
            import_btn = gr.Button("开始导入")
            import_out = gr.Markdown()
            embed_bar = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label="OpenAI嵌入进度", interactive=False)
            insert_bar = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label="Milvus写入进度", interactive=False)
            import_btn.click(
                fn=import_terms_with_progress,
                inputs=file_input,
                outputs=[import_out, embed_bar, insert_bar]
            )

if __name__ == "__main__":
    demo.launch() 