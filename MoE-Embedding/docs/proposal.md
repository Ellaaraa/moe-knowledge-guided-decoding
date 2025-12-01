Here is the converted Markdown file containing the full text from the uploaded PDF.

# [cite_start]Expert-Aware Knowledge-Guided Decoding [cite: 1]

[cite_start]**Authors:** Elaheh Rasoulian [cite: 1][cite_start], Neeraj Menon Suresh Kumar [cite: 2]

---

## [cite_start]1 Prior Work and Our Approach [cite: 3]

### [cite_start]1.1 Knowledge-Guided Decoding (KGD) [cite: 4]

[cite_start]The Knowledge-Guided Decoding (KGD) Burapacheep (2024) framework helps large language models generate more factually accurate text by directly influencing token selection during decoding[cite: 4]. Unlike Retrieval-Augmented Generation (RAG) systems Lewis et al. (2020)[cite_start], which retrieve external knowledge before decoding, KGD integrates retrieved information dynamically at each decoding step[cite: 5].

[cite_start]The model compares candidate tokens with the retrieved evidence and assigns higher probabilities to those aligned with factual content[cite: 6]. [cite_start]The rewards are based on semantic similarity - how closely the meaning of a token matches the fact retrieved and entailment, which is evaluated with a Natural Language Inference (NLI) model to determine whether the generated text logically follows from the evidence[cite: 7].

[cite_start]However, KGD has limitations[cite: 8]. [cite_start]Its performance depends on the weighting of reward signals: excessive weights can reduce fluency, while weak weights diminish factual gains[cite: 8]. [cite_start]It also struggles with generating accurate numbers and quantities, as existing reward functions capture broad semantic meaning but not numerical precision[cite: 9]. These limitations motivate our integration of Mixture-of-Experts (MoE) Shazeer et al. (2017) [cite_start]representations into the KGD framework[cite: 10].

### [cite_start]1.2 Mixture-of-Experts Embeddings (MOEE) [cite: 11]

[cite_start]In MoE LLMs, each layer's router produces a probability distribution over experts (a softmax over routing logits), selecting which experts process the input[cite: 11]. [cite_start]Treating these routing weights across layers as features and concatenating them yields an embedding that captures how the model internally "routes" an input through its experts[cite: 12].

[cite_start]Mixture-of-Experts (MoE) has been used primarily for scaling and efficiency - routing tokens to a small subset of experts to reduce compute - rather than for representation learning per se[cite: 13]. [cite_start]The MOEE paper Li and Zhou (2025) observes that the router's probability vector over experts at each layer (the "routing weights", RW) can be treated as an off-the-shelf embedding - a perspective underexplored by earlier MoE work[cite: 14].

[cite_start]The paper quantifies this via clustering metrics (e.g., low Jaccard and 45% exact-match across clusterings) and low Spearman correlations on STS12 [cite: 15, 16][cite_start]; it also reports RW is more robust to prompt variations than HS[cite: 17].

### [cite_start]1.3 What is new in our approach (KGD-MOEE with RW-alignment) [cite: 18, 19]

[cite_start]We replace the generic embedding in KGD's similarity reward with MOEE (HS+RW), leveraging the paper's finding that RW captures stable, high-level semantics that complement HS[cite: 19]. [cite_start]This aims to make the similarity reward more robust to prompt phrasing and more semantically faithful to the supporting evidence, while remaining training-free[cite: 20].

[cite_start]This RW-alignment reward is not present in prior KGD or MoE embedding work [cite: 21][cite_start]; it bridges MoE internals (routers) with guided decoding for factuality[cite: 22].

---

## [cite_start]2 Datasets [cite: 23]

### [cite_start]2.1 Natural Questions [cite: 24]

The Natural Questions dataset Kwiatkowski et al. (2019)[cite_start], released by Google Research, is a large-scale benchmark for open-domain question answering[cite: 25]. [cite_start]It contains real user queries from Google Search paired with corresponding Wikipedia articles that contain the answers[cite: 26]. [cite_start]Each example includes a question (an anonymized real-world search query), a Wikipedia page that may contain the answer, and human-annotated short answers (usually brief phrases or entities) and long answers (complete passages)[cite: 27].

### [cite_start]2.2 TriviaQA [cite: 28]

TriviaQA Joshi et al. (2017) [cite_start]pairs ~95k crowd-written trivia questions with independently gathered evidence pages (6 per question on average), yielding 650k+ question-answer-evidence triples across "Web" and "Wikipedia" domains[cite: 29]. [cite_start]We need a short-form, factoid QA benchmark where (i) retrieval matters and (ii) correctness can be scored with exact-match style metrics, matching the KGD setup (token-level rewards guided by retrieved evidence)[cite: 30].

---

## [cite_start]3 Baseline [cite: 31]

### [cite_start]3.1 Vanilla model [cite: 32]

[cite_start]Our first baseline is a vanilla language model[cite: 33].

### [cite_start]3.2 Knowledge-Guided Decoding (KGD) Model [cite: 34]

[cite_start]The second model we will implement is the KGD model, which builds on the same base language model but, as discussed in the literature review, modifies its decoding step to incorporate knowledge-guided rewards that adjust the model's token probabilities[cite: 35].

### [cite_start]3.3 MOEE-Enhanced KGD (HS+RW) - strong baseline for our method [cite: 36]

[cite_start]As a third baseline, we will replace the Hidden State embeddings with a Mixture-of-Experts-aware embedding (MOEE) that fuses the standard hidden-state similarity with router-weight (RW)[cite: 37].

---

## [cite_start]4 Analysis and Expected Results [cite: 38]

### [cite_start]4.1 Research questions [cite: 39]

1. [cite_start]Does MOEE (HS+RW) improve KGD's guidance over HS-only or generic embedders? [cite: 40]
2. [cite_start]Does the new RW-alignment reward reduce off-topic drift / hallucinations? [cite: 41]
3. [cite_start]How sensitive is the performance of our approach to the fusion weight (HSusRW) and guidance weight w? [cite: 42]
4. [cite_start]Where do gains come from better retrieval, better re-ranking, or better token-level steering? [cite: 43]

### [cite_start]4.2 Analysis [cite: 44]

- [cite_start]**Embedding source ablation (KGD-HS vs RW vs HS+RW/MOEE).** We evaluate how different embedding sources affect KGD's similarity reward by swapping the embedder used and measuring exact match (EM) and F1 performance[cite: 45].

  - [cite_start]**Expected outcome:** The combined embedding (HS+RW) should outperform HS-only and RW-only variants, particularly on paraphrased or semantically indirect questions, indicating improved semantic robustness[cite: 46].

- [cite_start]**Effect of the RW-alignment reward.** We test the impact of adding an RW-alignment term to KGD and compare hallucination rate, supported EM, and attribution@k under matched EM conditions[cite: 47].

  - [cite_start]**Expected outcome:** The alignment reward is expected to reduce hallucination and increase attribution accuracy without sacrificing fluency, while incurring only a modest latency overhead due to top-m scoring[cite: 48, 49].

- [cite_start]**Retrieval vs decoding contributions.** We evaluate whether performance gains arise from improved retrieval or decoding[cite: 50]. [cite_start]Retrieval quality is measured using nDCG and Recall@k before and after MOEE reranking, while decoding is analyzed through token-level reward changes and top-m inclusion rates[cite: 51].

  - [cite_start]**Expected outcome:** MOEE should yield better retrieval scores and more frequent inclusion of correct tokens among top-m candidates, indicating that both retrieval and token-level steering contribute to overall improvements[cite: 52].

- [cite_start]**Prompt robustness.** We also assess robustness to prompt variations by evaluating performance on a paraphrased development subset[cite: 53]. [cite_start]Accuracy variance across different prompt formulations is used as a stability indicator[cite: 54].
  - [cite_start]**Expected outcome:** The MOEE-enhanced model should exhibit lower accuracy variance than the HS-only baseline, demonstrating improved robustness to prompt rewording and higher consistency in factual responses[cite: 55].

### [cite_start]4.3 Expected Conclusion [cite: 56]

[cite_start]We expect the largest gains on MoE LLMs that expose router probabilities (RW available)[cite: 57]. [cite_start]Overall we expect, On TriviaQA, KGD-MOEE (HS+RW) will yield higher EM/F1 and Supported-EM, lower hallucination, and better prompt robustness than KGD-HS, with the clearest gains on MoE LLMs and questions requiring semantic reranking while keeping runtime overhead modest via the top-m approximation[cite: 58].

---

## [cite_start]5 LLMs for Experiment [cite: 59]

### [cite_start]5.1 OLMOE (Open Mixture-of-Experts Language Model) [cite: 60, 61]

OLMOE Muennighoff et al. (2025) [cite_start]is an open-source Mixture-of-Experts (MoE) language model[cite: 62]. [cite_start]We chose OLMOE for its open accessibility, efficient scaling, and ease of integration with our KGD framework for experimentation[cite: 62].

### [cite_start]5.2 DeepSeek-MoE [cite: 63]

DeepSeek-MoE Dai et al. (2024)[cite_start], we selected it for its efficient routing strategy, low compute cost, and suitability for testing KGD's generalizability[cite: 64].

---

## [cite_start]References [cite: 65]

- Jirayu Burapacheep. 2024. Enhancing factuality in language models through knowledge-guided decoding. [cite_start]Stanford CS224N Custom Project[cite: 66].
- [cite_start]Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang[cite: 67]. 2024. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. [cite_start]Preprint, arXiv:2401.06066[cite: 68].
- [cite_start]Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer[cite: 69]. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. [cite_start]Preprint, arXiv:1705.03551[cite: 70].
- [cite_start]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov[cite: 71]. 2019. Natural questions: A benchmark for question answering research. [cite_start]Transactions of the Association for Computational Linguistics, 7:452-466[cite: 72].
- [cite_start]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela[cite: 73]. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. [cite_start]In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS '20, Red Hook, NY, USA[cite: 74]. [cite_start]Curran Associates Inc[cite: 75].
- Ziyue Li and Tianyi Zhou. 2025. [cite_start]Your mixture-of-experts LLM is secretly an embedding model for free[cite: 76]. [cite_start]In The Thirteenth International Conference on Learning Representations[cite: 77].
- [cite_start]Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, and 5 others[cite: 78]. 2025. Olmoe: Open mixture-of-experts language models. [cite_start]Preprint, arXiv:2409.02060[cite: 79].
- [cite_start]Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean[cite: 80]. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. [cite_start]Preprint, arXiv: 1701.06538[cite: 81].
