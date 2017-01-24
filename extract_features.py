import numpy as np

def extract_features(predictor, img, d):
        shape = predictor(img, d)

        # Extract (x, y) coordinates of facial landmarks
        parts = [[shape.part(n).x, shape.part(n).y] for n in range(shape.num_parts)]
        parts = np.asarray(parts).astype(int)
        
        # Compute landmark coordinates with respect to position-in-frame
        # to enforce translation invariance of features (roughly, due to noise)
        parts_x = parts.T[0] - d.left()
        parts_y = parts.T[1] - d.top()

        return np.hstack((parts_x, parts_y)), parts
