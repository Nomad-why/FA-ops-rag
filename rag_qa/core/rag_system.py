# RAG系统流程整合： 意图识别 -> 策略选择 -> 文档检索 -> 答案生成

from rag_qa.core.prompts import RAGPrompts
import time
from base import config
from base import logger
from rag_qa.core.query_classifier import QueryClassifier  # 查询分类器
from rag_qa.core.strategy_selector import StrategySelector  # 策略选择器

#RAG系统类
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        # 向量数据库对象
        self.vector_store = vector_store
        # 大语言模型调用函数
        self.llm = llm
        # RAG的提示词模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 意图识别分类器
        self.query_classifier =QueryClassifier(model_path=f'{config.MODELS_DIR}/bert_query_classifier')
        # 检索策略选择器
        self.strategy_selector = StrategySelector()

    # 假设文档检索方法（HyDE）
    def _retrieve_with_hyde(self, query):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt() # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            # 使用假设答案进行检索，并返回检索结果
            # HyDE通常只用于生成检索向量，不一定需要rerank这一步，但hybrid_search方法中已包含rerank步骤，后续去掉的话需要修改。
            return self.vector_store.hybrid_search(
                hypo_answer, k=config.RETRIEVAL_K # 使用 K 而非 M
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []


    # 子查询检索方法
    def _retrieve_with_subqueries(self, query):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        # 子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt() # 使用 template 后缀区分
        try:
            # 调用llm生成子查询列表
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                 logger.warning("未能生成有效的子查询")
                 return []

            # 初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            # 遍历每个子查询
            for sub_q in subqueries:
                # 使用子查询进行检索，并将结果添加到列表中
                # 这里对每个子查询都执行了hybrid search，其中包含重排序，开销可能较大
                docs = self.vector_store.hybrid_search(
                    sub_q, k=config.RETRIEVAL_K # 使用 K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            # 对所有检索结果进行去重 (基于对象内存地址，如果Document内容相同但对象不同则无法去重)
            # 更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs} # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            # 返回去重后的文档，限制数量 (是否需要在此处限制? retrieve_and_merge 末尾会限制)
            return unique_docs # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    #  使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt() # 使用 template 后缀区分
        try:
            # 调用llm生成回溯问题
            simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            # 使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search(
                simplified_query, k=config.RETRIEVAL_K # 使用 K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    #   定义方法，检索并合并相关文档
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):  #   新增 strategy 参数
        #   如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        #   根据检索策略选择不同的检索方式
        ranked_sub_chunks = [] # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query) # 返回的是唯一文档列表
             # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
             # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:  #   默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search(
                query, k=config.RETRIEVAL_K, source_filter=source_filter
            ) # 注意 hybrid_search返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")
        final_context_docs = ranked_sub_chunks[:config.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    #   定义方法，生成答案
    def generate_answer(self, query, source_filter=None):
        # 记录查询开始时间
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")

        #  判断查询类型
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        # 如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            prompt_input = self.rag_prompt.format(
                context="", question=query, history=""
            )  #   不使用上下文
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服"
            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return answer

        #   否则，进行 RAG 检索并生成答案
        logger.info("查询为专业咨询，执行 RAG 流程")
        #   选择检索策略
        strategy = self.strategy_selector.select_strategy(query)

        #   检索相关文档
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )  #  传递strategy

        #   准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs]) # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")

        #   构造 Prompt，调用大语言模型生成答案
        prompt_input = self.rag_prompt.format(
            context=context, question=query
        )

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服"


        #   记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer

