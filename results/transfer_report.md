# Cross-game transfer: World of Tanks → Dota 2 (CONDA)

35,895 labelled Dota 2 messages from 1,917 matches, 19.5% toxic (explicit or implicit). No CONDA label is used for training or thresholds; the oracle column is only a reference.

AUROC CIs resample whole matches (cluster bootstrap).

| model                                  |   conda_auroc |   conda_auroc_ci_low |   conda_auroc_ci_high |   conda_f1_gametox_threshold |   conda_f1_oracle_threshold |   recall_explicit |   recall_implicit |   false_positive_rate |   gametox_test_auroc |
|:---------------------------------------|--------------:|---------------------:|----------------------:|-----------------------------:|----------------------------:|------------------:|------------------:|----------------------:|---------------------:|
| xlmr_base                              |         0.827 |                0.818 |                 0.836 |                        0.632 |                       0.639 |             0.847 |             0.114 |                 0.077 |                0.952 |
| xlmr_base_focal                        |         0.827 |                0.818 |                 0.836 |                        0.631 |                       0.643 |             0.863 |             0.126 |                 0.085 |                0.948 |
| xlmr_base_w075                         |         0.814 |                0.806 |                 0.824 |                        0.632 |                       0.634 |             0.842 |             0.108 |                 0.074 |                0.945 |
| char_bigru                             |         0.812 |                0.804 |                 0.822 |                        0.605 |                       0.610 |             0.786 |             0.100 |                 0.072 |                0.940 |
| generic_toxic_bert                     |         0.806 |                0.797 |                 0.815 |                        0.621 |                       0.623 |             0.811 |             0.118 |                 0.072 |                0.797 |
| char_cnn_focal                         |         0.794 |                0.785 |                 0.804 |                        0.599 |                       0.601 |             0.766 |             0.093 |                 0.068 |                0.933 |
| generic_multilingual_toxic_xlm_roberta |         0.786 |                0.777 |                 0.795 |                        0.571 |                       0.573 |             0.702 |             0.088 |                 0.062 |                0.835 |
| char_cnn                               |         0.776 |                0.766 |                 0.786 |                        0.569 |                       0.580 |             0.690 |             0.075 |                 0.055 |                0.935 |
| tfidf_lr                               |         0.756 |                0.744 |                 0.771 |                        0.604 |                       0.607 |             0.822 |             0.135 |                 0.092 |                0.941 |

`gametox_test_auroc` is the same binary task in the original game: the drop to `conda_auroc` measures how much is lost when moving to another game. A large gap between the GameTox-threshold F1 and the oracle F1 means the scores still rank messages well but the threshold does not transfer.
