python feat_prepare_for_av.py -input ./meta/meta.LSPEC.REC.4cls.av.txt -a_m_steps 100 -v_m_steps 5 -c_idx 2 -n_cc 23 -c_len 10 --three_d -mt 1:2:4:5:6:7 -out ../../features/ser/stl_vs_mtl/lstm/LSPEC.IMG.100.3d.cc.REC.4cls.av --no_image
python feat_prepare_for_av.py -input ./meta/meta.LSPEC.REC.4cls.txt -a_m_steps 100 -v_m_steps 5 -c_idx 2 -n_cc 23 -c_len 10 --three_d -mt 1:2:4:5:6:7 -out ../../features/ser/stl_vs_mtl/lstm/LSPEC.IMG.100.3d.cc.REC.4cls --no_image

python feat_prepare_for_av.py -input ./meta/meta.RAW.REC.4cls.av.txt -a_m_steps 16000 -v_m_steps 5 -c_idx 2 -n_cc 23 -c_len 1600 --two_d -mt 1:2:4:5:6:7 -out ../../features/ser/stl_vs_mtl/lstm/RAW.IMG.100.3d.cc.REC.4cls --no_image

python feat_prepare_for_av.py -input ./meta/meta.LSPEC.REC.4cls.av.txt -a_m_steps 100 -v_m_steps 10 -c_idx 2 -n_cc 23 -c_len 10 --three_d -mt 1:2:4:5:6:7 -out ../../features/ser/stl_vs_mtl/lstm/LSPEC.FIMG.100.10.3d.cc.REC.4cls.av

python feat_prepare_for_av.py -input ./meta/meta.LSPEC.REC.4cls.av.txt -a_m_steps 100 -v_m_steps 20 -c_idx 2 -n_cc 23 -c_len 10 --three_d -mt 1:2:4:5:6:7 -out ../../features/ser/stl_vs_mtl/lstm/LSPEC.FIMG.100.20.3d.cc.REC.4cls.av

python feat_prepare_for_av.py -input ./meta/meta.LSPEC.REC.4cls.b-av.txt -a_m_steps 100 -v_m_steps 20 -c_idx 2 -n_cc 23 -c_len 10 --three_d -mt 1:2:4:5:6:7 -out ../../features/ser/stl_vs_mtl/lstm/LSPEC.FIMG.100.20.3d.cc.REC.4cls.b-av

python feat_prepare_for_av.py -input ./meta/meta.RAW.REC.4cls.b-av.txt -a_m_steps 16000 -v_m_steps 20 -c_idx 2 -n_cc 23 -c_len 1600 -mt 1:2:4:5:6:7 -out ../../features/ser/stl_vs_mtl/lstm/RAW.FIMG.100.20.2d.cc.REC.4cls.b-av