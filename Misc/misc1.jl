img = testimage("mandril_color")

using ImageFeatures
keypoints_1 = Keypoints(fastcorners(Gray.(img), 12, 0.4))
brief_params = BRIEF(size = 256, window = 10, seed = 123)
desc_1, ret_keypoints_1 = create_descriptor((Gray.(img), keypoints_1, brief_params)