# Milestone 4 presentation: speaker script

About 4:50 in total, so there's a little room for pauses. Each person covers three slides. The same lines are in the speaker notes of `Milestone4_Slides.pptx`.

| Speaker | Slides |
|---|---|
| Jason | 1-3 |
| Cort | 4-6 |
| Saad | 7-9 |
| Adlene | 10-12 |

## Slide 1: Can First-Semester Data Flag Students at Risk of Dropping Out?
*Jason, about 15 s*

Hi everyone, we're Team 7. Our project asks whether a university could use the data it already has at the end of a student's first semester to flag students who might drop out.

## Slide 2: Introduction
*Jason, about 25 s*

Right now most schools notice a struggling student late, usually once their grades have already dropped. We use a public dataset of 4,424 students from a Portuguese polytechnic, and for each student we predict one of three outcomes: they drop out, they're still enrolled after the normal time, which means they're delayed, or they graduate.

## Slide 3: Research Questions
*Jason, about 20 s*

We had three questions. First, does first-semester data beat simple baselines by a clear margin? Second, does a non-linear model like kNN or a decision tree do better than our linear baseline? And third, can we catch more of the delayed Enrolled students without hurting the other classes?

## Slide 4: Data and Pipeline
*Cort, about 30 s*

We have 169 features after cleaning: admission information, demographics, finances and first-semester grades. We drop all the second-semester columns because they come after our prediction point and basically give the answer away. The classes aren't balanced: half the students graduate and only 18% are Enrolled. So our main metric is macro-F1, which counts all three classes equally, and we fixed it before training anything. We split 70/15/15, tune only on validation, and we haven't touched the test set yet.

## Slide 5: Methods
*Cort, about 25 s*

We compare three models. Our baseline from last milestone is softmax regression, which we wrote ourselves in NumPy and trained with gradient descent on the cross-entropy loss. The two new ones are kNN from Lab 3, which labels a student by their seven nearest neighbours, and a decision tree from Lab 4, which splits on whichever feature gives the most information gain. Each model is tuned on validation and then refit on ten different splits.

## Slide 6: RQ1 & RQ2: Model Comparison
*Cort, about 30 s*

Here are the results averaged over the ten splits. All three models are far above the majority-class baseline of 0.22, and above the 0.27 bar we set in Milestone 2, so the answer to the first question is yes. But for the second question the answer is no. The plain softmax model gets 0.667, the tree gets 0.624 and kNN 0.564, and softmax wins on nine of ten splits against the tree and all ten against kNN.

## Slide 7: Why Don't the Non-Linear Models Win?
*Saad, about 30 s*

So why don't the non-linear models win? The tree keeps fitting the training data better as it gets deeper, but past depth twelve validation stops improving, so the extra flexibility turns into overfitting. kNN memorizes the training set at k equals one and never gets past 0.55. We think that's because most of our 169 features are one-hot columns that are almost always zero, so every student ends up roughly the same distance from everyone else. And one feature, how many first-semester units a student passed, gives over a third of the tree's information gain, which suggests the signal is close to linear.

## Slide 8: Where the Models Fail
*Saad, about 20 s*

Here's where the models go wrong. Enrolled is the hardest class for all three. Softmax calls 36 Enrolled students Dropouts and 41 Graduates. kNN is worse in a way that matters more: it calls 81 real Dropouts Graduates, so those students would never get flagged.

## Slide 9: RQ3: Helping the Enrolled Class
*Saad, about 25 s*

For the third question we tried undersampling the training set, like in Lab 2, so every class has the same number of students. For softmax, Enrolled recall goes from 0.35 up to 0.61. The cost is that Graduate recall drops from 0.91 to 0.73 and macro-F1 drops a little, to 0.647. We'll pick between the two versions on validation in the next milestone.

## Slide 10: Preliminary Subgroup Check
*Adlene, about 25 s*

We also did a first subgroup check on the validation set. Two things stood out. For the 77 students behind on their tuition, the model predicts Dropout for every single one, so it's leaning heavily on tuition status. And for scholarship holders, it catches only 54% of the ones who drop out, compared with 72% for everyone else. Men and women get similar scores.

## Slide 11: Deviations, Blockers and Next Steps
*Adlene, about 25 s*

A few things changed from our plan. We moved the ensemble to Milestone 5, our best baseline turned out to depend on when gradient descent stops, and one of us ran the code on different library versions, though the results came out the same. Our main blockers are the Enrolled class and the tuition shortcut. Next we'll try bagged trees, fix the baseline, run the full subgroup analysis, and score the test set once on November 20th.

## Slide 12: Conclusion
*Adlene, about 20 s*

To wrap up: first-semester data can predict student outcomes well above chance, the simple linear model still beats kNN and a decision tree, and undersampling helps with delayed students at a small cost. Thanks for listening.
