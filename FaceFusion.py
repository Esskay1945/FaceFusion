from google.colab import files
uploaded = files.upload()import cv2


import cv2
import numpy as np
import mediapipe as mp
import os

class FaceAveragerARM:

    def __init__(self):
        print("🔧 Initializing face detector...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("✅ Ready to process faces!")

    def detect_landmarks(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        h, w = img.shape[:2]
        points = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        return np.array(points, dtype=np.float32)

    def get_key_points(self, landmarks):
        if landmarks is None:
            return None
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        return np.array([left_eye, right_eye], dtype=np.float32)

    def similarity_transform(self, in_pts, out_pts):
        s60 = np.sin(60 * np.pi / 180)
        c60 = np.cos(60 * np.pi / 180)
        in_pts = np.copy(in_pts).tolist()
        out_pts = np.copy(out_pts).tolist()
        xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
        yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]
        in_pts.append([xin, yin])
        xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
        yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]
        out_pts.append([xout, yout])
        tform = cv2.estimateAffinePartial2D(np.array(in_pts), np.array(out_pts))
        return tform[0]

    def normalize_face(self, img, landmarks, w=600, h=600):
        key_pts = self.get_key_points(landmarks)
        if key_pts is None:
            return None, None
        eye_left = np.array([0.3 * w, h / 3], dtype=np.float32)
        eye_right = np.array([0.7 * w, h / 3], dtype=np.float32)
        out_pts = np.array([eye_left, eye_right], dtype=np.float32)
        tform = self.similarity_transform(key_pts, out_pts)
        normalized_img = cv2.warpAffine(img, tform, (w, h))
        landmarks_homogeneous = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
        normalized_landmarks = np.dot(tform, landmarks_homogeneous.T).T
        return normalized_img, normalized_landmarks

    def calculate_delaunay_triangles(self, points, w=600, h=600):
        boundary_pts = [
            (0, 0), (w//2, 0), (w-1, 0),
            (w-1, h//2), (w-1, h-1),
            (w//2, h-1), (0, h-1), (0, h//2)
        ]
        points = np.vstack([points, boundary_pts])
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((float(p[0]), float(p[1])))
        triangle_list = subdiv.getTriangleList()
        triangles = []
        for t in triangle_list:
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            indices = []
            for pt in pts:
                for i, p in enumerate(points):
                    if abs(pt[0] - p[0]) < 1.0 and abs(pt[1] - p[1]) < 1.0:
                        indices.append(i)
                        break
            if len(indices) == 3:
                triangles.append(indices)
        return triangles, points

    def warp_triangle(self, img1, img2, t1, t2):
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        t1_rect = []
        t2_rect = []
        t2_rect_int = []
        for i in range(3):
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
        img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        size = (r2[2], r2[3])
        warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
        img2_rect = cv2.warpAffine(img1_rect, warp_mat, size, None,
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
        img2_rect = img2_rect * mask
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = \
            img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + img2_rect

    def average_faces(self, image_paths, output_path="average_face.jpg"):
        print(f"\n{'='*60}")
        print("🎭 STARTING FACE AVERAGING")
        print(f"{'='*60}\n")
        images = []
        all_landmarks = []
        print("📸 Processing images...")
        for i, img_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] Loading: {img_path}")
            if not os.path.exists(img_path):
                print(f"   ❌ File not found, skipping...")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"   ❌ Could not read image, skipping...")
                continue
            print(f"   🔍 Detecting face...")
            landmarks = self.detect_landmarks(img)
            if landmarks is None:
                print(f"   ❌ No face detected, skipping...")
                continue
            print(f"   ✅ Found {len(landmarks)} facial landmarks!")
            images.append(img)
            all_landmarks.append(landmarks)
        if len(images) == 0:
            raise ValueError("❌ No valid faces detected in any images!")
        print(f"\n✅ Successfully processed {len(images)} faces!")
        print("\n📐 Normalizing and aligning faces...")
        normalized_images = []
        normalized_landmarks = []
        for idx, (img, landmarks) in enumerate(zip(images, all_landmarks)):
            print(f"   [{idx+1}/{len(images)}] Aligning face...")
            result = self.normalize_face(img, landmarks)
            if result[0] is not None:
                normalized_images.append(result[0])
                normalized_landmarks.append(result[1])
        print("\n📊 Calculating average facial structure...")
        avg_landmarks = np.mean(normalized_landmarks, axis=0)
        print("🔺 Creating triangulation mesh...")
        triangles, points = self.calculate_delaunay_triangles(avg_landmarks)
        print(f"   Created {len(triangles)} triangles")
        print("\n🎨 Warping and blending faces...")
        output = np.zeros(normalized_images[0].shape, dtype=np.float32)
        boundary_pts = [
            (0, 0), (300, 0), (599, 0),
            (599, 300), (599, 599),
            (300, 599), (0, 599), (0, 300)
        ]
        for idx, (img, landmarks) in enumerate(zip(normalized_images, normalized_landmarks)):
            print(f"   [{idx+1}/{len(normalized_images)}] Processing face...")
            warped = np.zeros(img.shape, dtype=np.float32)
            landmarks_with_boundary = np.vstack([landmarks, boundary_pts])
            for triangle in triangles:
                t1 = [landmarks_with_boundary[triangle[0]],
                      landmarks_with_boundary[triangle[1]],
                      landmarks_with_boundary[triangle[2]]]
                t2 = [points[triangle[0]],
                      points[triangle[1]],
                      points[triangle[2]]]
                self.warp_triangle(img, warped, t1, t2)
            output = output + warped
        output = output / len(normalized_images)
        output = np.clip(output, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, output)
        print(f"\n{'='*60}")
        print(f"🎉 SUCCESS!")
        print(f"{'='*60}")
        print(f"💾 Saved to: {os.path.abspath(output_path)}")
        print(f"📊 Averaged {len(normalized_images)} faces")
        print(f"{'='*60}\n")
        return output


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          FACE AVERAGING - ARM WINDOWS VERSION            ║
    ║              Works on Snapdragon Processors!             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    try:
        averager = FaceAveragerARM()
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        print("\n💡 Make sure you installed: pip install opencv-python mediapipe numpy")
        exit(1)
    image_paths = [
        "1.jpg",
        "3.jpg",
    ]
    print("\n📋 Image files to process:")
    for i, path in enumerate(image_paths, 1):
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {i}. {path} {exists}")
    try:
        result = averager.average_faces(image_paths, "my_average_face.jpg")
        print("\n🖼️  Displaying result...")
        cv2.imshow("Average Face", result)
        print("   Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tips:")
        print("   - Make sure image file paths are correct")
        print("   - Images should contain clear, front-facing faces")
        print("   - Try with different images if faces aren't detected")



        from google.colab.patches import cv2_imshow
import cv2

result = cv2.imread("/content/my_average_face.jpg")
cv2_imshow(result)