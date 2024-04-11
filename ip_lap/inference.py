import os,cv2,torch,subprocess,platform
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from .draw_landmark import draw_landmarks
import face_alignment
from .face_mask import FaceMask
from cuda_malloc import cuda_malloc_supported
from .models import Landmark_generator as Landmark_transformer,Renderer,audio

NAME = "IP_LAP"

# the following is the index sequence for fical landmarks detected by mediapipe
ori_sequence_idx = [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
                    361, 323, 454, 356, 389,  #
                    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  #
                    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,  #
                    168, 6, 197, 195, 5,  #
                    48, 115, 220, 45, 4, 275, 440, 344, 278,  #
                    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  #
                    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,  #
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  #
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# the following is the connections of landmarks for drawing sketch image
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])
FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_CONNECTION = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])

full_face_landmark_sequence = [*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),  #upper-half face
                               *list(range(4, 21)),  # jaw
                               *list(range(91, 131))]  # mouth

class LandmarkDict(dict):# Makes a dictionary that behave like an object to represent each landmark
    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

class IP_LAP_infer:

    def __init__(self,T=5,Nl=15,ref_img_N=25,
                 img_size=128,mel_step_size=16,
                 face_det_batch_size=4,
                 checkpoints_path=""):
        self.T = T
        self.Nl = Nl
        self.ref_img_N = ref_img_N
        self.img_size = img_size
        self.mel_step_size = mel_step_size
        self.face_det_batch_size = face_det_batch_size
        self.pads = [100,100,100,100]
        self.device = "cuda" if cuda_malloc_supported() else "cpu"

        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.lip_index = [0, 17] 
        self.all_landmarks_idx = self.summarize_landmark(FACEMESH_CONNECTION)
        self.pose_landmark_idx = \
            self.summarize_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE,
                                                    FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
                [162, 127, 234, 93, 389, 356, 454, 323])
        # pose landmarks are landmarks of the upper-half face(eyes,nose,cheek) that represents the pose information

        self.content_landmark_idx = self.all_landmarks_idx - self.pose_landmark_idx
        # content_landmark include landmarks of lip and jaw which are inferred from audio


        landmark_gen_checkpoint_path = os.path.join(checkpoints_path, "landmarkgenerator_checkpoint.pth")
        renderer_checkpoint_path = os.path.join(checkpoints_path, "renderer_checkpoint.pth")
        self.landmark_generator_model = self.load_model(
                model=Landmark_transformer(T=self.T, d_model=512, nlayers=4, nhead=4, dim_feedforward=1024, dropout=0.1),
                path=landmark_gen_checkpoint_path)
        self.renderer = self.load_model(model=Renderer(), path=renderer_checkpoint_path)

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)

        self.face_mask = FaceMask()

    def __call__(self,video_file, audio_file, outfile):
        temp_dir = os.path.join(os.path.dirname(outfile), NAME)
        if not os.path.exists(temp_dir): os.makedirs(temp_dir, exist_ok=True)
        ##(1) Reading input video frames  ###
        print(f'[Step 1]Reading video frames ... from {video_file}', NAME)
        if not os.path.isfile(video_file):
            raise ValueError('the input video file does not exist')
        elif video_file.split('.')[1] in ['jpg', 'png', 'jpeg']: #if input a single image for testing
            ori_background_frames = [cv2.imread(video_file)]
        else:
            video_stream = cv2.VideoCapture(video_file)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            if fps != 25:
                print(" input video fps:", fps,',converting to 25fps...')
                tmp_file = '{}/temp_25fps.mp4'.format(temp_dir)
                if os.path.exists(tmp_file): os.remove(tmp_file)
                print(tmp_file)
                command = 'ffmpeg -y -i ' + video_file + f' -r 25 {tmp_file}'
                subprocess.call(command, shell=platform.system() != 'Windows',stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                video_file = '{}/temp_25fps.mp4'.format(temp_dir)
                video_stream.release()
                video_stream = cv2.VideoCapture(video_file)
                fps = video_stream.get(cv2.CAP_PROP_FPS)
            assert fps == 25

            ori_background_frames = [] #input videos frames (includes background as well as face)
            frame_idx = 0
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                ori_background_frames.append(frame)
                frame_idx = frame_idx + 1
        input_vid_len = len(ori_background_frames)

        ##(2) Extracting audio####
        print(f'[Step 2]Extracting audio ... from {audio_file}', NAME)
        if not audio_file.endswith('.wav'):
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_file, '{}/temp.wav'.format(temp_dir))
            subprocess.call(command, shell=platform.system() != 'Windows', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            audio_file = '{}/temp.wav'.format(temp_dir)
        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)  # (H,W)   extract mel-spectrum
        ##read audio mel into list###
        mel_chunks = []  # each mel chunk correspond to 5 video frames, used to generate one video frame
        mel_idx_multiplier = 80. / fps
        mel_chunk_idx = 0
        while 1:
            start_idx = int(mel_chunk_idx * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])  # mel for generate one video frame
            mel_chunk_idx += 1
        
        print('[Step 3]detect facial using face detection tool', NAME)
        ori_face_frames, ori_face_coords = self.face_detect(ori_background_frames)
        # print(len(ori_face_frames))
        import gc; gc.collect(); torch.cuda.empty_cache()

        ##(3) detect facial landmarks using mediapipe tool
        print('[Step 4]detect facial landmarks using mediapipe tool', NAME)
        boxes = []  #bounding boxes of human face
        lip_dists = [] #lip dists
        #we define the lip dist(openness): distance between the  midpoints of the upper lip and lower lip
        face_crop_results = []
        all_pose_landmarks, all_content_landmarks = [], []  #content landmarks include lip and jaw landmarks
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0) as face_mesh:
            # (1) get bounding boxes and lip dist
            for frame_idx, full_frame in tqdm(enumerate(ori_face_frames),total=input_vid_len,
                                              desc="get bounding boxes and lip dist"):
                h, w = full_frame.shape[0], full_frame.shape[1]
                results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
                if not results.multi_face_landmarks:
                    raise NotImplementedError  # not detect face
                face_landmarks = results.multi_face_landmarks[0]

                ## calculate the lip dist
                dx = face_landmarks.landmark[self.lip_index[0]].x - face_landmarks.landmark[self.lip_index[1]].x
                dy = face_landmarks.landmark[self.lip_index[0]].y - face_landmarks.landmark[self.lip_index[1]].y
                dist = np.linalg.norm((dx, dy))
                lip_dists.append((frame_idx, dist))

                # (1)get the marginal landmarks to crop face
                x_min,x_max,y_min,y_max = 999,-999,999,-999
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in self.all_landmarks_idx:
                        if landmark.x < x_min:
                            x_min = landmark.x
                        if landmark.x > x_max:
                            x_max = landmark.x
                        if landmark.y < y_min:
                            y_min = landmark.y
                        if landmark.y > y_max:
                            y_max = landmark.y
                ##########plus some pixel to the marginal region##########
                #note:the landmarks coordinates returned by mediapipe range 0~1
                plus_pixel = 25
                x_min = max(x_min - plus_pixel / w, 0)
                x_max = min(x_max + plus_pixel / w, 1)

                y_min = max(y_min - plus_pixel / h, 0)
                y_max = min(y_max + plus_pixel / h, 1)
                y1, y2, x1, x2 = int(y_min * h), int(y_max * h), int(x_min * w), int(x_max * w)
                boxes.append([y1, y2, x1, x2])
            boxes = np.array(boxes)

            # (2)croppd face
            face_crop_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] \
                                for image, (y1, y2, x1, x2) in zip(ori_face_frames, boxes)]

            # (3)detect facial landmarks
            for frame_idx, full_frame in tqdm(enumerate(ori_face_frames),total=input_vid_len,
                                              desc="detect facial landmarks"):
                h, w = full_frame.shape[0], full_frame.shape[1]
                results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
                if not results.multi_face_landmarks:
                    raise ValueError("not detect face in some frame!")  # not detect
                face_landmarks = results.multi_face_landmarks[0]



                pose_landmarks, content_landmarks = [], []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in self.pose_landmark_idx:
                        pose_landmarks.append((idx, w * landmark.x, h * landmark.y))
                    if idx in self.content_landmark_idx:
                        content_landmarks.append((idx, w * landmark.x, h * landmark.y))

                # normalize landmarks to 0~1
                y_min, y_max, x_min, x_max = face_crop_results[frame_idx][1]  #bounding boxes
                pose_landmarks = [ \
                    [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in pose_landmarks]
                content_landmarks = [ \
                    [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in content_landmarks]
                all_pose_landmarks.append(pose_landmarks)
                all_content_landmarks.append(content_landmarks)
        
        all_pose_landmarks = self.get_smoothened_landmarks(all_pose_landmarks, windows_T=1)
        all_content_landmarks=self.get_smoothened_landmarks(all_content_landmarks,windows_T=1)

        ##randomly select N_l reference landmarks for landmark transformer##
        print("randomly select N_l reference landmarks for landmark transformer", NAME)
        dists_sorted = sorted(lip_dists, key=lambda x: x[1])
        lip_dist_idx = np.asarray([idx for idx, dist in dists_sorted])  #the frame idxs sorted by lip openness

        Nl_idxs = [lip_dist_idx[int(i)] for i in torch.linspace(0, input_vid_len - 1, steps=self.Nl)]
        Nl_pose_landmarks, Nl_content_landmarks = [], []  #Nl_pose + Nl_content=Nl reference landmarks
        for reference_idx in Nl_idxs:
            frame_pose_landmarks = all_pose_landmarks[reference_idx]
            frame_content_landmarks = all_content_landmarks[reference_idx]
            Nl_pose_landmarks.append(frame_pose_landmarks)
            Nl_content_landmarks.append(frame_content_landmarks)

        Nl_pose = torch.zeros((self.Nl, 2, 74))  # 74 landmark
        Nl_content = torch.zeros((self.Nl, 2, 57))  # 57 landmark
        for idx in range(self.Nl):
            #arrange the landmark in a certain order, since the landmark index returned by mediapipe is is chaotic
            Nl_pose_landmarks[idx] = sorted(Nl_pose_landmarks[idx],
                                            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
            Nl_content_landmarks[idx] = sorted(Nl_content_landmarks[idx],
                                            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

            Nl_pose[idx, 0, :] = torch.FloatTensor(
                [Nl_pose_landmarks[idx][i][1] for i in range(len(Nl_pose_landmarks[idx]))])  # x
            Nl_pose[idx, 1, :] = torch.FloatTensor(
                [Nl_pose_landmarks[idx][i][2] for i in range(len(Nl_pose_landmarks[idx]))])  # y
            Nl_content[idx, 0, :] = torch.FloatTensor(
                [Nl_content_landmarks[idx][i][1] for i in range(len(Nl_content_landmarks[idx]))])  # x
            Nl_content[idx, 1, :] = torch.FloatTensor(
                [Nl_content_landmarks[idx][i][2] for i in range(len(Nl_content_landmarks[idx]))])  # y
        Nl_content = Nl_content.unsqueeze(0)  # (1,Nl, 2, 57)
        Nl_pose = Nl_pose.unsqueeze(0)  # (1,Nl,2,74)


        ##select reference images and draw sketches for rendering according to lip openness##
        print("select reference images and draw sketches for rendering according to lip openness", NAME)
        ref_img_idx = [int(lip_dist_idx[int(i)]) for i in torch.linspace(0, input_vid_len - 1, steps=self.ref_img_N)]
        ref_imgs = [face_crop_results[idx][0] for idx in ref_img_idx]
        ## (N,H,W,3)
        ref_img_pose_landmarks, ref_img_content_landmarks = [], []
        for idx in ref_img_idx:
            ref_img_pose_landmarks.append(all_pose_landmarks[idx])
            ref_img_content_landmarks.append(all_content_landmarks[idx])

        ref_img_pose = torch.zeros((self.ref_img_N, 2, 74))  # 74 landmark
        ref_img_content = torch.zeros((self.ref_img_N, 2, 57))  # 57 landmark

        for idx in range(self.ref_img_N):
            ref_img_pose_landmarks[idx] = sorted(ref_img_pose_landmarks[idx],
                                                key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
            ref_img_content_landmarks[idx] = sorted(ref_img_content_landmarks[idx],
                                                    key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
            ref_img_pose[idx, 0, :] = torch.FloatTensor(
                [ref_img_pose_landmarks[idx][i][1] for i in range(len(ref_img_pose_landmarks[idx]))])  # x
            ref_img_pose[idx, 1, :] = torch.FloatTensor(
                [ref_img_pose_landmarks[idx][i][2] for i in range(len(ref_img_pose_landmarks[idx]))])  # y

            ref_img_content[idx, 0, :] = torch.FloatTensor(
                [ref_img_content_landmarks[idx][i][1] for i in range(len(ref_img_content_landmarks[idx]))])  # x
            ref_img_content[idx, 1, :] = torch.FloatTensor(
                [ref_img_content_landmarks[idx][i][2] for i in range(len(ref_img_content_landmarks[idx]))])  # y

        ref_img_full_face_landmarks = torch.cat([ref_img_pose, ref_img_content], dim=2).cpu().numpy()  # (N,2,131)
        ref_img_sketches = []
        for frame_idx in range(ref_img_full_face_landmarks.shape[0]):  # N
            full_landmarks = ref_img_full_face_landmarks[frame_idx]  # (2,131)
            h, w = ref_imgs[frame_idx].shape[0], ref_imgs[frame_idx].shape[1]
            drawn_sketech = np.zeros((int(h * self.img_size / min(h, w)), int(w * self.img_size / min(h, w)), 3))
            mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]], full_landmarks[0, idx],
                                                    full_landmarks[1, idx]) for idx in range(full_landmarks.shape[1])]
            drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                        connection_drawing_spec=self.drawing_spec)
            drawn_sketech = cv2.resize(drawn_sketech, (self.img_size, self.img_size))  # (128, 128, 3)
            ref_img_sketches.append(drawn_sketech)
        ref_img_sketches = torch.FloatTensor(np.asarray(ref_img_sketches) / 255.0).cuda().unsqueeze(0).permute(0, 1, 4, 2, 3)
        # (1,N, 3, 128, 128)
        ref_imgs = [cv2.resize(face.copy(), (self.img_size, self.img_size)) for face in ref_imgs]
        ref_imgs = torch.FloatTensor(np.asarray(ref_imgs) / 255.0).unsqueeze(0).permute(0, 1, 4, 2, 3).cuda()
        # (1,N,3,H,W)

        ##prepare output video strame##
        frame_h, frame_w = ori_background_frames[0].shape[:-1]
        '''
        out_stream = cv2.VideoWriter('{}/result.avi'.format(temp_dir), cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                    (frame_w, frame_h))  # +frame_h*3
        '''
        out_stream = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                    (frame_w, frame_h))  # +frame_h*3

        ##generate final face image and output video##
        input_mel_chunks_len = len(mel_chunks)
        input_frame_sequence = torch.arange(input_vid_len).tolist()
        #the input template video may be shorter than audio
        #in this case we repeat the input template video as following
        num_of_repeat=input_mel_chunks_len//input_vid_len+1
        input_frame_sequence = input_frame_sequence + list(reversed(input_frame_sequence))
        input_frame_sequence=input_frame_sequence*((num_of_repeat+1)//2)
        file_num = 0
        for batch_idx, batch_start_idx in tqdm(enumerate(range(0, input_mel_chunks_len-2, 1)),
                                       total=len(range(0, input_mel_chunks_len-2, 1)), desc="[IP_LAP] [Step 5]Lipsync..."):
            T_input_frame, T_ori_face_coordinates = [], []
            #note: input_frame include background as well as face
            T_mel_batch, T_crop_face,T_pose_landmarks = [], [],[]

            A_input_frame, A_ori_face_coordinates = [], []

            # (1) for each batch of T frame, generate corresponding landmarks using landmark generator
            for mel_chunk_idx in range(batch_start_idx, batch_start_idx + self.T):  # for each T frame
                # 1 input audio
                T_mel_batch.append(mel_chunks[max(0, mel_chunk_idx - 2)])

                # 2.input face
                input_frame_idx = int(input_frame_sequence[mel_chunk_idx])
                face, coords = face_crop_results[input_frame_idx]
                T_crop_face.append(face)
                T_ori_face_coordinates.append((face, coords))  ##input face
                # 3.pose landmarks
                T_pose_landmarks.append(all_pose_landmarks[input_frame_idx])
                # 3.face background
                T_input_frame.append(ori_face_frames[input_frame_idx].copy())
                # 4.frame background
                A_ori_face_coordinates.append(ori_face_coords[input_frame_idx])
                A_input_frame.append(ori_background_frames[input_frame_idx].copy())

            T_mels = torch.FloatTensor(np.asarray(T_mel_batch)).unsqueeze(1).unsqueeze(0)  # 1,T,1,h,w
            #prepare pose landmarks
            T_pose = torch.zeros((self.T, 2, 74))  # 74 landmark
            for idx in range(self.T):
                T_pose_landmarks[idx] = sorted(T_pose_landmarks[idx],
                                            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
                T_pose[idx, 0, :] = torch.FloatTensor(
                    [T_pose_landmarks[idx][i][1] for i in range(len(T_pose_landmarks[idx]))])  # x
                T_pose[idx, 1, :] = torch.FloatTensor(
                    [T_pose_landmarks[idx][i][2] for i in range(len(T_pose_landmarks[idx]))])  # y
            T_pose = T_pose.unsqueeze(0)  # (1,T, 2,74)

            #landmark  generator inference
            Nl_pose, Nl_content = Nl_pose.cuda(), Nl_content.cuda() # (Nl,2,74)  (Nl,2,57)
            T_mels, T_pose = T_mels.cuda(), T_pose.cuda()
            with torch.no_grad():  # require    (1,T,1,hv,wv)(1,T,2,74)(1,T,2,57)
                predict_content = self.landmark_generator_model(T_mels, T_pose, Nl_pose, Nl_content)  # (1*T,2,57)
            T_pose = torch.cat([T_pose[i] for i in range(T_pose.size(0))], dim=0)  # (1*T,2,74)
            T_predict_full_landmarks = torch.cat([T_pose, predict_content], dim=2).cpu().numpy()  # (1*T,2,131)

            #1.draw target sketch
            T_target_sketches = []
            for frame_idx in range(self.T):
                full_landmarks = T_predict_full_landmarks[frame_idx]  # (2,131)
                h, w = T_crop_face[frame_idx].shape[0], T_crop_face[frame_idx].shape[1]
                drawn_sketech = np.zeros((int(h * self.img_size / min(h, w)), int(w * self.img_size / min(h, w)), 3))
                mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]]
                                                        , full_landmarks[0, idx], full_landmarks[1, idx]) for idx in
                                            range(full_landmarks.shape[1])]
                drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                            connection_drawing_spec=self.drawing_spec)
                drawn_sketech = cv2.resize(drawn_sketech, (self.img_size, self.img_size))  # (128, 128, 3)
                if frame_idx == 2:
                    show_sketch = cv2.resize(drawn_sketech, (frame_w, frame_h)).astype(np.uint8)
                T_target_sketches.append(torch.FloatTensor(drawn_sketech) / 255)
            T_target_sketches = torch.stack(T_target_sketches, dim=0).permute(0, 3, 1, 2)  # (T,3,128, 128)
            target_sketches = T_target_sketches.unsqueeze(0).cuda()  # (1,T,3,128, 128)

            # 2.lower-half masked face
            ori_face_img = torch.FloatTensor(cv2.resize(T_crop_face[2], (self.img_size, self.img_size)) / 255).permute(2, 0, 1).unsqueeze(
                0).unsqueeze(0).cuda()  #(1,1,3,H, W)

            # 3. render the full face
            # require (1,1,3,H,W)   (1,T,3,H,W)  (1,N,3,H,W)   (1,N,3,H,W)  (1,1,1,h,w)
            # return  (1,3,H,W)
            with torch.no_grad():
                generated_face, _, _, _ = self.renderer(ori_face_img, target_sketches, ref_imgs, ref_img_sketches,
                                                            T_mels[:, 2].unsqueeze(0))  # T=1
            gen_face = (generated_face.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H,W,3)

            # 4. paste each generated face
            y1, y2, x1, x2 = T_ori_face_coordinates[2][1]  # coordinates of face bounding box
            original_background = T_input_frame[2].copy()
            T_input_frame[2][y1:y2, x1:x2] = cv2.resize(gen_face,(x2 - x1, y2 - y1))  #resize and paste generated face
            # 5. post-process
            full_face = self.merge_face_contour_only(original_background, T_input_frame[2], T_ori_face_coordinates[2][1],self.fa)   #(H,W,3)
            # 6.output
            # full = np.concatenate([show_sketch, full], axis=1)
            # print(f"full_face.shape{full_face.shape}")
            ori_x1, ori_y1, ori_x2, ori_y2 = A_ori_face_coordinates[2]
            # print(ori_face_coords[file_num])
            full_frame = A_input_frame[2]
            #full_mask = np.zeros_like(full_frame)
            # print(full_frame.shape)
            if ori_x1 != -1:
                p = cv2.resize(full_face.astype(np.uint8), (ori_x2 - ori_x1, ori_y2 - ori_y1))
                # print(p.shape)
                full_frame[ori_y1:ori_y2, ori_x1:ori_x2] = p
                # height, width = full_frame.shape[:2]
                # img = self.Laplacian_Pyramid_Blending_with_mask(full_frame, ori_background_frames[file_num], full_mask[:, :, 0], 6)
                # pp = np.uint8(cv2.resize(np.clip(img, 0 ,255), (width, height)))
                mask = self.face_mask(p)
                full_frame[ori_y1:ori_y2, ori_x1:ori_x2] = full_frame[ori_y1:ori_y2, ori_x1:ori_x2] * (1 - mask[..., None]) + p * mask[..., None]
            full = full_frame.copy()

            out_stream.write(full)
            
            try:
               # cv2.imwrite(temp_frame_paths[batch_idx+2],full)
                file_num += 1
            except:
                pass
            
            if batch_idx == 0:
                out_stream.write(full)
                out_stream.write(full)
                # cv2.imwrite(temp_frame_paths[batch_idx],full)
                # cv2.imwrite(temp_frame_paths[batch_idx+1],full)
                
        out_stream.release()
        # command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(voice_file, '{}/result.avi'.format(temp_dir), outfile)
        # subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"succeed output results to:{outfile}", NAME)


    def face_detect(self, images):
        
        batch_size = self.face_det_batch_size
        
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    imgs = np.array(images[i:i + batch_size])
                    imgs_numpy = imgs.transpose(0, 3, 1, 2)
                    image_batch = torch.from_numpy(imgs_numpy.copy())
                    _, _, bboxes =self.fa.get_landmarks_from_batch(image_batch,return_bboxes=True)
                    predictions.extend(bboxes)
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):

            if rect is None:
                # cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                # results.append([-1,-1,-1,-1])
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            else:
                rect = rect[0]
                rect = np.clip(rect, 0, None)
                x1_0, y1_0, x2_0, y2_0 = map(int, rect[:-1])

                y1 = max(0, y1_0 - pady1)
                y2 = min(image.shape[0], y2_0 + pady2)
                x1 = max(0, x1_0 - padx1)
                x2 = min(image.shape[1], x2_0 + padx2)
            
                results.append([x1, y1, x2, y2])

        boxes = np.array(results)

        faces = [image[y1: y2, x1:x2] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        return faces, boxes


    def merge_face_contour_only(self,src_frame, generated_frame, face_region_coord, fa): #function used in post-process
        """Merge the face from generated_frame into src_frame
        """
        input_img = src_frame
        y1, y2, x1, x2 = 0, 0, 0, 0
        if face_region_coord is not None:
            y1, y2, x1, x2 = face_region_coord
            input_img = src_frame[y1:y2, x1:x2]
        ### 1) Detect the facial landmarks
        try:
            preds = fa.get_landmarks(input_img)[0]  # 68x2
        except:
            preds = np.int64(-1 * np.ones((68,2)))
        if face_region_coord is not None:
            preds += np.array([x1, y1])
        lm_pts = preds.astype(int)
        contour_idx = list(range(0, 17)) + list(range(17, 27))[::-1]
        contour_pts = lm_pts[contour_idx]
        ### 2) Make the landmark region mark image
        mask_img = np.zeros((src_frame.shape[0], src_frame.shape[1], 1), np.uint8)
        cv2.fillConvexPoly(mask_img, contour_pts, 255)
        ### 3) Do swap
        img = self.swap_masked_region(src_frame, generated_frame, mask=mask_img)
        return img

    def swap_masked_region(self,target_img, src_img, mask): #function used in post-process
        """From src_img crop masked region to replace corresponding masked region
        in target_img
        """  # swap_masked_region(src_frame, generated_frame, mask=mask_img)
        mask_img = cv2.GaussianBlur(mask, (21, 21), 11)
        mask1 = mask_img / 255
        mask1 = np.tile(np.expand_dims(mask1, axis=2), (1, 1, 3))
        img = src_img * mask1 + target_img * (1 - mask1)
        return img.astype(np.uint8)

    # smooth landmarks
    def get_smoothened_landmarks(self,all_landmarks, windows_T=1):
        for i in range(len(all_landmarks)):  # frame i
            if i + windows_T > len(all_landmarks):
                window = all_landmarks[len(all_landmarks) - windows_T:]
            else:
                window = all_landmarks[i: i + windows_T]
            #####
            for j in range(len(all_landmarks[i])):  # landmark j
                all_landmarks[i][j][1] = np.mean([frame_landmarks[j][1] for frame_landmarks in window])  # x
                all_landmarks[i][j][2] = np.mean([frame_landmarks[j][2] for frame_landmarks in window])  # y
        return all_landmarks
    
    def load_model(self, model, path):
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            if k[:6] == 'module':
                new_k=k.replace('module.', '', 1)
            else:
                new_k =k
            new_s[new_k] = v
        model.load_state_dict(new_s)
        model = model.to(self.device)
        return model.eval()
    
    def _load(self,checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint
    
    def summarize_landmark(self, edge_set):  # summarize all ficial landmarks used to construct edge
        landmarks = set()
        for a, b in edge_set:
            landmarks.add(a)
            landmarks.add(b)
        return landmarks