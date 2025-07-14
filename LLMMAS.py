from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS, DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import json
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

class configs():
    openai_api_key= "sk-no-key-required"
    # set for llama.cpp server
    openai_api_base = "http://127.0.0.1:8080/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    client.api_base="http://127.0.0.1:8080/v1"
    embedding_model = "all-MiniLM-L6-v2"
    llm_model = "gemma3:1b"
    # path to the reference paper that describes the semantic communications
    pdf_path = "1.pdf"
    # path to the reference code that is as an example
    code_path = "code.txt"
    # iteration number
    iterations = 3
    debug = True

# Define LLM-enhanced multi-agent system
class LLM_enhanced_multi_agent_system_for_SC():
    def __init__(self, args: configs):
        self.api_key = args.openai_api_key
        self.api_base = args.openai_api_base
        self.llm = ChatOpenAI(model_name=args.llm_model, temperature=0.9, api_key=self.api_key, base_url=self.api_base)
        self.iterations = args.iterations
        self.embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
        self.client = args.client
        self.llm_model_name = args.llm_model
        self.debug_mode = args.debug

    class ModeratorResponse(BaseModel):
        flagged: bool = Field(description="Set to true if the query contains harmful content. Otherwise false.")
    
    # Validate the legitimacy of the input
    def secure_agent(self, query):
        
        system_prompt = f"""
                            You are a content moderation agent. Your task is to check if the user query contains any harmful content.
                            """
        response = self.client.beta.chat.completions.parse(
            model=self.llm_model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role":"user", "content": f"Here is the query: {query}"}],
            response_format=self.ModeratorResponse
        )
        print(response)
        moderation_output = response.choices[0].message.parsed
        print(moderation_output)
        if moderation_output.flagged:
            print("Illegal input!")
            return False
        else:
            return query

    # Load the paper and generate vector memory, which is the knowledge base
    def condensate_agent(self, pdf_path=""):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # construct knowledge base
        self.db = DocArrayInMemorySearch.from_documents(
            docs,
            self.embeddings
        )

    # Distill relevant SC knowledge for constructing the SC model
    def inference_agent(self, query):
        retriever = self.db.as_retriever()
        docs = retriever.get_relevant_documents(query)
    
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following documents to answer the question.
    
                {context}
    
                Question: {question}
                Answer:"""
        )
    
        doc_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
    
        context = "\n\n".join(doc.page_content for doc in docs)
        final_prompt = prompt.format(context=context, question=query)

        if self.debug_mode:
            print("\nFinal Prompt Sent to LLM:\n")
            print(final_prompt)
        
        context = [Document(page_content=doc.page_content) for doc in docs]
    
        answer = doc_chain.invoke({"question":query, "context": context})    
        if self.debug_mode:
            print("\nLLM Answer:\n", answer)
        return answer

    # Formulates a specific sub-task chain for the SC model
    def planning_agent(self, input_name="paper_ref", output_name="sc_scheme"):
        scheme_prompt = ChatPromptTemplate.from_template(
            "# Referring to the following description of the semantic communication model: '''{"
            f"{input_name}"
            "}'''."
            "# Output the design scheme for achieving the semantic communication model."
        )
        chain = LLMChain(llm=self.llm, prompt=scheme_prompt,
                         output_key=output_name,
                         )
        return chain

    # sub-task chain for code generation
    def code_generation(self, input_name="sc_scheme", output_name="sc_model"):
        data = [
            "Hello there!",
            "How are you doing?",
            "PyTorch is great.",
            "Machine learning is fascinating.",
            "Let's build something amazing.",
            "Semantic communication matters.",
            "Understanding context is important.",
            "AI is changing the world.",
            "Keep learning and growing.",
            "Innovation drives progress."
        ]
        model_prompt = ChatPromptTemplate.from_template(
            "# According to the design scheme of the semantic communication model: '''{"
            f"{input_name}"
            "}'''."
            "# Refering to the reference code: '''{ref_code}''',"
            "output the Pytorch framework-based Python code for achieving the semantic communication model, "
            f"assuming the input data is '''{data}''', which is a list of sentences. "
            f"Specifically, the Additive Gaussian White Noise (AWGN) can serve as the physical channel. The Bilingual Evaluation Understudy (BLEU) score is adopted as the metric that evaluates the SC model. We expect that the generated SC model achieves no less than a 0.6 BLEU score when the Signal-to-Noise Ratio (SNR) is 10 dB. The total number of model parameters does not exceed 2,000,000 for the resource constraints of devices."
        )
        chain = LLMChain(llm=self.llm, prompt=model_prompt,
                         output_key=output_name,
                         )
        return chain

    # evaluate the quality of the generated code
    def evaluate_agent(self, input_name="sc_model", output_name="eval_score"):
        # Manually input bleu scores for SC model
        eval_prompt = ChatPromptTemplate.from_template(
            "The quality of the generated code is evaluated and an overall score is given from three aspects: "
            "1. Quality of code, including its reasonableness and completeness."
            "2. The performance of code, using bleu score as the evaluation indicator."
            "3. Whether the number of parameters in the generated code meets the limitation."
            "Based on the above requirements, evaluate the quality of the 'generated code':'''{"
            f"{input_name}"
            "}'''."
            "## Output the evaluated results. The format of the output refers to:"
            "```json Dict('model parameters': int \ the model parameters of the 'generated code', 'evaluated score': int \ The evaluated score is based on the performance of the reference code in three aspects. The evaluated score ranges from 0 to 100 and is not the same as the last time."
            "'evaluation': string \ Based on the performance of the reference code in three aspects, give the reviews, including the model parameters, the description of 'generated code', the advantage and disadvantage analysis of the 'generated code', etc."
            "There is an example: 'The Python-based SC model has been successfully implemented, incorporating all necessary modules. The semantic encoder and decoder are realized using Long Short-Term Memory (LSTM) networks. The channel encoder and decoder are constructed based on the Multilayer Perceptron (MLP) architecture.  The final SC model can achieve a 0.68 BLEU score when SNR is 10 dB, which meets expectations. In addition, the total number of model parameters is 1,826,762.')```"
        )
        chain = LLMChain(llm=self.llm, prompt=eval_prompt,
                         output_key=f"{output_name}",
                         )
        return chain

    def long_memory_stroage(self, inputs, output_name="long_term_memory"):
        memory_prompt = ChatPromptTemplate.from_template(
            "# According to the code and evaluation results of the current subtask chain:'''{"
            f"{inputs[0]}"
            "}''' and '''{"
            f"{inputs[1]}"
            "}'''."
            "# If there is a significant difference in the semantic space of the current subtask chain, i.e. the 'evaluation score' is less than 60."
            "# Then only the module composition of the sub task chain code needs to be output, such as LSTM, MLP, etc."
        )
        chain = LLMChain(llm=self.llm, prompt=memory_prompt,
                         output_key=f"{output_name}",
                         )
        return chain

    def short_memory_stroage(self, inputs, output_name="short_term_memory"):
        memory_prompt = ChatPromptTemplate.from_template(
            "# According to the code and evaluation results of the current subtask chain:'''{"
            f"{inputs[0]}"
            "}''' and '''{"
            f"{inputs[1]}"
            "}'''."
            "# If the current subtask chain has semantic similarity in the semantic space, i.e. the 'evaluation score' is greater than 60."
            "# Not only does it output the module composition of the subtask chain code, such as LSTM, MLP, etc., but it also outputs the evaluation results of the subtask chain"
        )
        chain = LLMChain(llm=self.llm, prompt=memory_prompt,
                         output_key=f"{output_name}",
                         )
        return chain

    def reflexion_agent(self, inputs, output_name="sc_model"):
        model_prompt = ChatPromptTemplate.from_template(
            "# Modify the Python code: '''{"
            f"{inputs[0]}"
            "}'''."
            "# According to the corresponding evaluation results:'''{"
            f"{inputs[1]}"
            "}'''."
            "# According to short-term memory:'''{"
            f"{inputs[2]}"
            "}'''."
            "# Extracting fine-grained information, similar to how humans can recall recent details, considering the performance of the current subtask chain in historical schemes, provides valuable small feedback for improving code."
            "# Output the modified Python codes that aim to improve the 'evaluated score' and obtain a score of no less than 90 finally."
        )
        chain = LLMChain(llm=self.llm, prompt=model_prompt,
                         output_key=output_name,
                         )
        return chain

    def refinement_agent(self, inputs, output_name="sc_model"):
        model_prompt = ChatPromptTemplate.from_template(
            "# Modify the Python code: '''{"
            f"{inputs[0]}"
            "}'''."
            "# According to the corresponding evaluation results:'''{"
            f"{inputs[1]}"
            "}'''."
            "# According to long-term memory:'''{"
            f"{inputs[2]}"
            "}'''."
            "# Quoting coarse-grained information is similar to the way humans extract important experiences from long-term decisions, considering the performance of the current subtask chain from a global perspective and providing large-scale feedback for improving subtask chain code."
            "# Output the modified Python codes that aim to improve the 'evaluated score' and obtain a score of no less than 90 finally."
        )
        chain = LLMChain(llm=self.llm, prompt=model_prompt,
                         output_key=output_name,
                         )
        return chain

    # Combine all agents for SC system generation
    def create_chains(self):
        chains = []
        output_keys = []

        # define sc scheme 1
        input_name = "paper_ref"
        output_name = "scheme_1"
        chain = self.planning_agent(input_name, output_name)
        chains.append(chain)
        output_keys.append(output_name)
        # define sc scheme 2
        input_name = "paper_ref"
        output_name = "scheme_2"
        chain = self.planning_agent(input_name, output_name)
        chains.append(chain)
        output_keys.append(output_name)
        # define code generation 1
        input_name = "scheme_1"
        output_name = "sc_model_0_1"
        chain = self.code_generation(input_name, output_name)
        chains.append(chain)
        output_keys.append(chain.output_key)

        # define code generation 2
        input_name = "scheme_2"
        output_name = "sc_model_0_2"
        chain = self.code_generation(input_name, output_name)
        chains.append(chain)
        output_keys.append(chain.output_key)
        for i in range(self.iterations):
            # eval sc model
            input_name = f"sc_model_{i}_1"
            output_name = f"eval_score_{i}_1"
            chain = self.evaluate_agent(input_name, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            input_name = f"sc_model_{i}_2"
            output_name = f"eval_score_{i}_2"
            chain = self.evaluate_agent(input_name, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            # memory storage
            inputs = [f"sc_model_{i}_1", f"eval_score_{i}_1"]
            output_name = f"long_term_memory_{i}_1"
            chain = self.long_memory_stroage(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            inputs = [f"sc_model_{i}_1", f"eval_score_{i}_1"]
            output_name = f"short_term_memory_{i}_1"
            chain = self.short_memory_stroage(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            inputs = [f"sc_model_{i}_2", f"eval_score_{i}_2"]
            output_name = f"long_term_memory_{i}_2"
            chain = self.long_memory_stroage(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            inputs = [f"sc_model_{i}_2", f"eval_score_{i}_2"]
            output_name = f"short_term_memory_{i}_2"
            chain = self.short_memory_stroage(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            # Reflect and modify the SC model according to the evaluated results and memory
            inputs = [f"sc_model_{i}_1", f"eval_score_{i}_1", f"short_term_memory_{i}_1"]
            output_name = f"sc_modelx_{i}_1"
            chain = self.reflexion_agent(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            inputs = [f"sc_model_{i}_2", f"eval_score_{i}_2", f"short_term_memory_{i}_2"]
            output_name = f"sc_modelx_{i}_2"
            chain = self.reflexion_agent(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            inputs = [f"sc_modelx_{i}_1", f"eval_score_{i}_1", f"long_term_memory_{i}_1"]
            output_name = f"sc_model_{i + 1}_1"
            chain = self.refinement_agent(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

            inputs = [f"sc_modelx_{i}_2", f"eval_score_{i}_2", f"long_term_memory_{i}_2"]
            output_name = f"sc_model_{i + 1}_2"
            chain = self.refinement_agent(inputs, output_name)
            chains.append(chain)
            output_keys.append(chain.output_key)

        # eval the final sc model
        input_name = f"sc_model_{self.iterations}_1"
        output_name = f"eval_score_{self.iterations}_1"
        chain = self.evaluate_agent(input_name, output_name)
        chains.append(chain)
        output_keys.append(chain.output_key)

        input_name = f"sc_model_{self.iterations}_2"
        output_name = f"eval_score_{self.iterations}_2"
        chain = self.evaluate_agent(input_name, output_name)
        chains.append(chain)
        output_keys.append(chain.output_key)

        self.overall_chain = SequentialChain(
            chains=chains,
            input_variables=["paper_ref", "ref_code"],
            output_variables=output_keys,
            verbose=True,
        )

    # Run multi-agent system
    def Run(self, inputs):
        outputs = self.overall_chain(inputs)
        return outputs

# generate sc by LLM enhanced multi-agent system
args = configs()
ref_code = open(args.code_path,"r",encoding="utf-8").read()
LL_SC = LLM_enhanced_multi_agent_system_for_SC(args)
LL_SC.condensate_agent(args.pdf_path)
paper_query = "Composition of the semantic communication model"
if LL_SC.secure_agent(paper_query):
    SC_Components = LL_SC.inference_agent(paper_query)
    LL_SC.create_chains()
    results = LL_SC.Run({"paper_ref":SC_Components,"ref_code":ref_code})
else:
    pass

print(results)