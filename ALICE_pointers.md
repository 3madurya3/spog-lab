# Using ALICE for SPOG research!

We are using the model that can be found in this repository -> https://github.com/orasanen/ALICE (steps listed therein are useful to go through!)

Installing it -> https://github.com/orasanen/ALICE/blob/new_diarizer/docs/installation.md

Running ALICE and understanding output -> https://github.com/orasanen/ALICE/blob/new_diarizer/docs/usage.md


*Copying the following snippet from the main ALICE directory to highlight it:*

In addition, utterance-level outputs for detected adult speech can be found from ALICE_output_utterances.txt, where each row corresponds to one utterance detected by the diarizer together with its estimated phoneme, syllable, and word counts. Timestamps appended to the filenames are of form <onset_time_in_ms x 10> _ <offset_time_in_ms x 10>, as measured from the beginning of each audio file. For instance, <filename>_0000062740_0000096150.wav stands for an utterance in <filename.wav> that started at 6.274 seconds and ended at 9.615 seconds.

NOTE: We need to make a few tweaks to a few scripts so that CHI and KCHI are also included (the original scripts only diarize MAL & FEM)

These can be found in the folder 'changes_to_ALICE'
