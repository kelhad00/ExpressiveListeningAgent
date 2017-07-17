#best result for arousal and valence so far
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --valence --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/

#best result for liking so far
python ./emotion_baseline_scripts/run_baseline.py -delay 2.2 --liking --text -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/


#PCM: do not work, LDA requires codebooks for classification
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ --pca -pl_dim 2000

python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ --pca -pl_dim 1000

python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ --pca -pl_dim 500

python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ --lda -pl_dim 2000

python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ --lda -pl_dim 1000

python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ --lda -pl_dim 500

python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ --spca

#just for visualise features
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --audio -path_audio ./audio_features_functionals_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ -path_save_train_feat ./tsne/alldfunc.train.total.csv -path_save_devel_feat ./tsne/alldfunc.devel.total.csv
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --video -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ -path_save_train_feat ./tsne/vbox.train.total.csv -path_save_devel_feat ./tsne/vbox.devel.total.csv
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --audio -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ -path_save_train_feat ./tsne/abox.train.total.csv -path_save_devel_feat ./tsne/abox.devel.total.csv
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --text -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ -path_save_train_feat ./tsne/tbox.train.total.csv -path_save_devel_feat ./tsne/tbox.devel.total.csv
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ -path_save_train_feat ./tsne/abox.vbox.tbox.train.total.csv -path_save_devel_feat ./tsne/abox.vbox.tbox.devel.total.csv

#sparse PCA and decision tree
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ -path_save_train_feat ./tsne/abox.vbox.tbox.train.spca.csv -path_save_devel_feat ./tsne/abox.vbox.tbox.devel.spca.csv --spca --dc_tree

#use pre-saved (dimension reduced) features
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --path_train ./analysis/temp_feat/abox.vbox.tbox.train.spca.csv --path_devel ./analysis/temp_feat/abox.vbox.tbox.devel.spca.csv

#kernel PCA and decision tree
python ./emotion_baseline_scripts/run_baseline.py -delay 1.2 --arousal --text --audio --video -path_audio ./audio_features_xbow_6s/ -path_video ./video_features_xbow_6s/ -path_text ./text_features_xbow_6s/ -path_save_train_feat ./tsne/abox.vbox.tbox.train.kpca.csv -path_save_devel_feat ./tsne/abox.vbox.tbox.devel.kpca.csv -path_save_test_feat ./tsne/abox.vbox.tbox.test.kpca.csv --kpca
