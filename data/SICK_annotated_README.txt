:::::::::::::::::::::::::::: University of Trento - Italy :::::::::::::::::::::::::::::::::::::

::::::::::::::::: SICK (Sentences Involving Compositional Knowledge) data set :::::::::::::::::

::::::::::::::::::::::::::http://clic.cimec.unitn.it/composes/sick/ :::::::::::::::::::::::::::


1) INTRODUCTION

The SICK data set consists of 9,840 English sentence pairs, built starting from two existing sets: 
the 8K ImageFlickr data (http://nlp.cs.illinois.edu/HockenmaierGroup/data.html) 
and the  SemEval 2012 STS MSR-Video Description data set 
(http://www.cs.york.ac.uk/semeval-2012/task6/index.php?id=data).

In order to generate SICK sentence pairs, we randomly selected a subset of sentence pairs from each 
source data set and we applied a 3-step process. First, the original sentences were
"normalized" to remove unwanted linguistic phenomena (S1). Each of the two normalized sentences S1
were then "expanded" to obtain up to three new sentences (S2,S3,S4) for each S1. 
As a third step, all the sentences generated in the expansion phase were "paired" to the two 
normalized sentences. More precisely, each S1 in the pair was combined with all the sentences 
resulting from the expansion phase (S2,S3,S4) and with the other S1 in the pair.

Finally, each sentence pair was annotated for relatedness in meaning and for the entailment relation 
between the two elements.

All the details about data set creation and annotation are given in the LREC paper included in this release.

The SICK data set was used in SemEval 2014 - Task 1: Evaluation of compositional distributional 
semantic models on full sentences through semantic relatedness and textual entailment
All the details about the task are given in the Task Overview paper included in this release.


2) LICENSE AND PUBLICATION CREDIT

The SICK data set is released under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 
Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US)

When using SICK in published research, please cite:
M. Marelli, S. Menini, M. Baroni, L. Bentivogli, R. Bernardi and R. Zamparelli. 2014. 
A SICK cure for the evaluation of compositional distributional semantic models. 
Proceedings of LREC 2014, Reykjavik (Iceland): ELRA. 



3) DATA STRUCTURE

This data set release contains the annotation of the "expansion rules" followed to create S2,S3,S4 
sentences from the respective S1 sentence in the pair.

File Structure: tab-separated text file

Fields:

- pair_ID: sentence pair ID

- pair type:
   - SnSn_intra: S2,S3, and S4 are paired with the S1 from which they were created through the 
     application of expansion rules.
   - SnSn_inter: S2,S3, and S4 are paired with the other S1 in the original pair
   - unrel: sentences from different original pairs are randomly paired

- sentence_A: sentence A

- sentenceA_expRule: annotation of sentence A. The tag includes the type of sentence (S1,S2,S3,S4) 
and the expansion rule followed to create the sentence. The rule is annotated only when it applies, 
namely for S2,S3,S4 sentences of pair type "_intra". The expansion rules and their corresponding 
labels are listed at the end of this file. All the other sentences are annotated as "Sn_null".

- sentence_B: sentence B

- sentenceB_expRule: annotation of sentence B. The tag includes the type of sentence (S1,S2,S3,S4) 
and the expansion rule followed to create the sentence. The rule is annotated only when it applies, 
namely for S2,S3,S4 sentences of pair type "_intra". The expansion rules and their corresponding 
labels are listed at the end of this file. All the other sentences are annotated as "Sn_null".

- entailment_label: textual entailment gold label (NEUTRAL, ENTAILMENT, or CONTRADICTION)

- relatedness_score: semantic relatedness gold score (on a 1-5 continuous scale)

- entailment_AB: entailment for the A-B order (A_neutral_B, A_entails_B, or A_contradicts_B)

- entailment_BA: entailment for the B-A order (B_neutral_A, B_entails_A, or B_contradicts_A)

- sentence_A_original: original sentence from which sentence A is derived

- sentence_B_original: original sentence from which sentence B is derived

- sentence_A_dataset: dataset from which the original sentence A was extracted (FLICKR vs. SEMEVAL)

- sentence_B_dataset: dataset from which the original sentence B was extracted (FLICKR vs. SEMEVAL)

- SemEval_set: set including the sentence pair in SemEval 2014 Task 1 (TRIAL, TRAIN, or TEST)


===

Annotation of Expansion rules to obtain S2: 
- toa (Turn passive sentences into active), 17 instances
- top (Turn active sentences into passive), 281 instances
- lex (Replace words with synonyms), 847 instances
- aa (Add modifiers), 287 instances
- expn (Expand agentive nouns), 28 instances
- expc (Turn compounds into relative clauses), 56 instances
- expa (Turn adjectives into relative clauses), 189 instances
- det (Replace quantifiers), 268 instances

Annotation of Expansion rules to obtain S3: 
- inv (Insert a negation), 419 instances
- od (Change determiners with opposites), 608 instances
- so (Replace words with semantic opposites), 933 instances

Annotation of Expansion rules to obtain S4: 
- ws (Scramble words), 377 instances

NOTE that the number of annotations for each expansion rule is different from that reported in 
the LREC paper (Table 4). The figures in the LREC paper refer to the original data set before the 
removal of 160 problematic pairs, while the figures above refer to the released data set.
