import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from PIL import Image
import base64
import io

class UpperBodyDetector:
    def __init__(self, overlay_image_path="assets/person_outline.png"):
        # Load YOLO model for person detection
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Initialize MediaPipe Pose for detailed body part detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load overlay template
        self.load_overlay_template(overlay_image_path)
        
        # Upper body landmarks indices (MediaPipe)
        self.upper_body_landmarks = [
            11, 12,  # Shoulders
            13, 14,  # Elbows
            15, 16,  # Wrists
            23, 24,  # Hips (for upper body boundary)
            0,       # Nose (head reference)
        ]
        
        # Minimum required landmarks for upper body detection
        self.required_landmarks = [11, 12, 23, 24, 0]  # Shoulders, hips, nose
        
    def load_overlay_template(self, overlay_path):
        """Load and process overlay template image"""
        try:
            self.overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
            if self.overlay_img is None:
                raise ValueError(f"Could not load overlay image from {overlay_path}")
            
            # Extract overlay dimensions and create mask
            if self.overlay_img.shape[2] == 4:
                alpha = self.overlay_img[:, :, 3]
                _, overlay_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
            else:
                gray_overlay = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGR2GRAY)
                _, overlay_mask = cv2.threshold(gray_overlay, 10, 255, cv2.THRESH_BINARY)
            
            # Find overlay contours to get dimensions
            contours, _ = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                self.overlay_width, self.overlay_height = w, h
                
                # Calculate upper body ratio (assuming upper body is top 60% of template)
                self.upper_body_height_ratio = 0.6
                self.expected_upper_body_height = int(h * self.upper_body_height_ratio)
            else:
                self.overlay_width, self.overlay_height = 200, 400
                self.expected_upper_body_height = 240
                
        except Exception as e:
            print(f"Error loading overlay: {e}")
            # Default values
            self.overlay_width, self.overlay_height = 200, 400
            self.expected_upper_body_height = 240

    def process_frame_from_base64(self, base64_image):
        """Process frame received as base64 string from frontend"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image.split(',')[1] if ',' in base64_image else base64_image)
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return self.process_frame(frame)
        except Exception as e:
            return {"error": f"Failed to process base64 image: {str(e)}"}

    def process_frame(self, frame):
        """Main processing function for each frame"""
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate expected overlay dimensions for this frame
        scale_factor = min(frame_width / self.overlay_img.shape[1], 
                          frame_height / self.overlay_img.shape[0]) * 0.9
        
        expected_width = int(self.overlay_width * scale_factor)
        expected_height = int(self.overlay_height * scale_factor)
        expected_upper_height = int(self.expected_upper_body_height * scale_factor)
        
        # Center overlay position
        overlay_x = (frame_width - expected_width) // 2
        overlay_y = (frame_height - expected_height) // 2
        
        # Draw overlay guide
        frame_with_overlay = self.draw_overlay_guide(frame, overlay_x, overlay_y, 
                                                   expected_width, expected_height)
        
        # Detect person using YOLO
        person_detected = False
        person_box = None
        
        yolo_results = self.yolo_model(frame)
        for result in yolo_results:
            for box in result.boxes:
                if result.names[int(box.cls)] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_box = (x1, y1, x2, y2)
                    person_detected = True
                    break
            if person_detected:
                break
        
        if not person_detected:
            return {
                "status": "no_person",
                "message": "Please step into the frame",
                "frame": self.frame_to_base64(frame_with_overlay),
                "upper_body_complete": False
            }
        
        # Analyze upper body completeness using MediaPipe
        upper_body_analysis = self.analyze_upper_body_completeness(frame, person_box)
        
        # Check positioning against overlay
        positioning_result = self.check_upper_body_positioning(
            person_box, upper_body_analysis, 
            overlay_x, overlay_y, expected_width, expected_upper_height
        )
        
        # Draw results on frame
        annotated_frame = self.draw_detection_results(
            frame_with_overlay, person_box, upper_body_analysis, positioning_result
        )
        
        return {
            "status": positioning_result["status"],
            "message": positioning_result["message"],
            "frame": self.frame_to_base64(annotated_frame),
            "upper_body_complete": upper_body_analysis["complete"],
            "positioning_score": positioning_result["score"],
            "details": {
                "visible_landmarks": upper_body_analysis["visible_count"],
                "total_landmarks": len(self.required_landmarks),
                "height_match": positioning_result["height_match"],
                "position_match": positioning_result["position_match"]
            }
        }

    def analyze_upper_body_completeness(self, frame, person_box):
        """Analyze if complete upper body is visible using MediaPipe Pose"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return {
                    "complete": False,
                    "visible_count": 0,
                    "landmarks": {},
                    "upper_body_box": None
                }
            
            # Extract landmark coordinates
            landmarks = {}
            visible_count = 0
            frame_height, frame_width = frame.shape[:2]
            
            for idx in self.upper_body_landmarks:
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    if landmark.visibility > 0.5:  # Visible landmark
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        landmarks[idx] = (x, y)
                        if idx in self.required_landmarks:
                            visible_count += 1
            
            # Calculate upper body bounding box from visible landmarks
            upper_body_box = None
            if landmarks:
                x_coords = [pos[0] for pos in landmarks.values()]
                y_coords = [pos[1] for pos in landmarks.values()]
                upper_body_box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            # Check if upper body is complete
            required_visible = visible_count >= len(self.required_landmarks) * 0.8  # 80% of required landmarks
            
            return {
                "complete": required_visible,
                "visible_count": visible_count,
                "landmarks": landmarks,
                "upper_body_box": upper_body_box
            }
            
        except Exception as e:
            print(f"Error in pose analysis: {e}")
            return {
                "complete": False,
                "visible_count": 0,
                "landmarks": {},
                "upper_body_box": None
            }

    def check_upper_body_positioning(self, person_box, upper_body_analysis, 
                                   overlay_x, overlay_y, expected_width, expected_upper_height):
        """Check if upper body positioning matches the overlay requirements"""
        
        if not upper_body_analysis["complete"]:
            return {
                "status": "incomplete_upper_body",
                "message": "Please ensure your complete upper body is visible",
                "score": 0.0,
                "height_match": False,
                "position_match": False
            }
        
        upper_body_box = upper_body_analysis["upper_body_box"]
        if upper_body_box is None:
            upper_body_box = person_box
        
        ub_x1, ub_y1, ub_x2, ub_y2 = upper_body_box
        upper_body_height = ub_y2 - ub_y1
        upper_body_width = ub_x2 - ub_x1
        
        # Calculate center positions
        person_center_x = (ub_x1 + ub_x2) / 2
        person_center_y = (ub_y1 + ub_y2) / 2
        overlay_center_x = overlay_x + expected_width / 2
        overlay_center_y = overlay_y + expected_upper_height / 2
        
        # Height matching (upper body should be at least 75% of expected height)
        height_ratio = upper_body_height / expected_upper_height
        height_match = height_ratio >= 0.75
        
        # Position matching (center should be close to overlay center)
        center_distance = np.sqrt((person_center_x - overlay_center_x)**2 +
                                (person_center_y - overlay_center_y)**2)
        max_allowed_distance = min(expected_width, expected_upper_height) * 0.25
        position_match = center_distance <= max_allowed_distance
        
        # Width check (should be reasonable compared to expected)
        width_ratio = upper_body_width / expected_width
        width_match = 0.5 <= width_ratio <= 1.5
        
        # Calculate overall score
        score = 0.0
        if height_match:
            score += 0.4
        if position_match:
            score += 0.4
        if width_match:
            score += 0.2
        
        print(f"[Debug Positioning] Height Ratio: {height_ratio:.2f} (Match: {height_match})")
        print(f"[Debug Positioning] Center Distance: {center_distance:.2f} (Max Allowed: {max_allowed_distance:.2f}, Match: {position_match})")
        print(f"[Debug Positioning] Width Ratio: {width_ratio:.2f} (Match: {width_match})")
        print(f"[Debug Positioning] Final Score: {score:.2f}")
        
        # Determine status and message
        if score >= 0.8:
            status = "perfect_position"
            message = "Perfect! Upper body positioned correctly"
        elif score >= 0.7:
            status = "good_position"
            message = "Good position, minor adjustments needed"
        elif not height_match:
            if height_ratio < 0.7:
                status = "too_far"
                message = "Please move closer - upper body too small"
            else:
                status = "too_close"
                message = "Please step back - upper body too large"
        elif not position_match:
            status = "wrong_position"
            message = "Please center yourself in the frame"
        else:
            status = "adjust_position"
            message = "Please adjust your position"
        
        return {
            "status": status,
            "message": message,
            "score": score,
            "height_match": height_match,
            "position_match": position_match,
            "height_ratio": height_ratio,
            "center_distance": center_distance
        }

    def draw_overlay_guide(self, frame, x, y, width, height):
        """Draw semi-transparent overlay guide"""
        overlay = frame.copy()
        
        # Draw filled rectangle with transparency
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), -1)
        
        # Draw upper body section (top 60%)
        upper_height = int(height * 0.6)
        cv2.rectangle(overlay, (x, y), (x + width, y + upper_height), (0, 255, 255), -1)
        
        # Blend with original frame
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
        cv2.rectangle(frame, (x, y), (x + width, y + upper_height), (0, 255, 255), 2)
        
        # Add text labels
        cv2.putText(frame, "Full Body Guide", (x, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Upper Body Focus", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame

    def draw_detection_results(self, frame, person_box, upper_body_analysis, positioning_result):
        """Draw detection results on frame"""
        if person_box:
            # Choose color based on positioning result
            if positioning_result["status"] == "perfect_position":
                color = (0, 255, 0)  # Green
            elif positioning_result["status"] == "good_position":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw person bounding box
            x1, y1, x2, y2 = person_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw upper body box if available
            if upper_body_analysis["upper_body_box"]:
                ub_x1, ub_y1, ub_x2, ub_y2 = upper_body_analysis["upper_body_box"]
                cv2.rectangle(frame, (ub_x1, ub_y1), (ub_x2, ub_y2), (255, 0, 255), 2)
            
            # Draw pose landmarks
            for landmark_id, (lx, ly) in upper_body_analysis["landmarks"].items():
                cv2.circle(frame, (lx, ly), 5, (255, 255, 0), -1)
            
            # Add status text
            cv2.putText(frame, positioning_result["message"], (x1, y1 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add score
            score_text = f"Score: {positioning_result['score']:.2f}"
            cv2.putText(frame, score_text, (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add landmark count
            landmark_text = f"Landmarks: {upper_body_analysis['visible_count']}/{len(self.required_landmarks)}"
            cv2.putText(frame, landmark_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame

    def frame_to_base64(self, frame):
        """Convert frame to base64 for frontend"""
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"

# Usage example
def process_image_from_frontend(base64_image, detector=None):
    """Function to be called from your API endpoint"""
    if detector is None:
        detector = UpperBodyDetector()
    
    result = detector.process_frame_from_base64(base64_image)
    return result

# Example usage for testing
if __name__ == "__main__":
    # Initialize detector
    detector = UpperBodyDetector()
    
    # Test with webcam (for development)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.process_frame(frame)
        print(f"Status: {result['status']}, Message: {result['message']}")
        
        # You would typically send result['frame'] back to frontend
        # For testing, we'll just display it
        if 'error' not in result:
            # Decode base64 for display
            img_data = base64.b64decode(result['frame'].split(',')[1])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            display_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            cv2.imshow('Upper Body Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()