from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from llama_index import Document as LlamaDocument
from pdfminer.high_level import extract_text


class DocSummarize:
    text_splitter = CharacterTextSplitter()

    def __init__(self, text: str) -> None:
        self.texts = self.text_splitter.split_text(text)

    def summarize(self):
        pass


class LangChainSummarize(DocSummarize):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.docs = [Document(page_content=t) for t in self.texts]


class MapReduceSummarize(LangChainSummarize):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.llm = OpenAI(temperature=0)
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            return_intermediate_steps=True
        )

    def summarize(self):
        summary = self.chain({"input_documents": self.docs},
                             return_only_outputs=True)

        intermediate_steps = '\n'.join(
            [step.strip("\n") for step in summary['intermediate_steps']])
        output_text = summary['output_text'].strip("\n")
        return f"""Summary: {output_text}, \n\nKey Points: {intermediate_steps}"""


class StuffSummarize(DocSummarize):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.llm = OpenAI(temperature=0)
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type='stuff',
            prompt=PromptTemplate(
                template="Write a concise summary of the following and list the key points\n\n{text}\n\n",
                input_variables=["text"]
            )
        )

    def summarize(self):
        summary = self.chain.run(self.docs)
        return f"""Summary: {summary}"""


class LlamaSummarize(DocSummarize):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
        prompt_helper = PromptHelper(
            max_input_size=4096,
            num_output=512,
            max_chunk_overlap=20
        )
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            prompt_helper=prompt_helper
        )

        self.docs = [LlamaDocument(t) for t in self.texts]
        self.index = GPTSimpleVectorIndex.from_documents(
            self.docs, service_context=service_context)

    def summarize(self):
        summary = self.index.query(
            """To summarise the paper, the summary needs to include these aspects: 
                1. the background to the paper
                2. the relevant research
                3. the methods used
                4. the result of the paper
            Then list a few key points.
            Finally give a few possible questions about the paper and provide answers.""",
            response_mode="tree_summarize")
        return f"""Summary: {summary.response}"""


def main():
    text = extract_text("demo.pdf")
    doc_summarize = LlamaSummarize(text)
    print(doc_summarize.summarize())


if __name__ == "__main__":
    main()
