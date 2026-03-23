#!/usr/bin/env python3

import sys
import rospy
import tf2_ros
import geometry_msgs.msg
import yaml
import argparse
import numpy as np
import threading
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QPushButton, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from tf.transformations import quaternion_matrix, quaternion_from_matrix

def transform_to_mat44(t):
    q = t.transform.rotation
    R = quaternion_matrix([q.x, q.y, q.z, q.w])
    R[0, 3] = t.transform.translation.x
    R[1, 3] = t.transform.translation.y
    R[2, 3] = t.transform.translation.z
    return R

def quaternion_to_axis_angle(qw, qx, qy, qz):
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    qw = np.clip(qw, -1.0, 1.0)
    angle = 2.0 * np.arccos(qw)
    sin_half_angle = np.sin(angle / 2.0)
    if abs(sin_half_angle) > 1e-6:
        axis = np.array([qx, qy, qz]) / sin_half_angle
    else:
        axis = np.array([qx, qy, qz])
        axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm if axis_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    return axis * angle

class TFGui(QWidget):
    def __init__(self, args):
        super().__init__()

        rospy.init_node("dynamic_tf_broadcaster", anonymous=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()

        self.scale = 100.0  # Increased scale for smoother slider mapping
        self.world_frame = 'world' 
        self.temp_frame = 'base_link' # camera base
        self._last_middle_T = None
        
        if args.load:
            with open(args.load, "r") as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                args.frame_id = data["cam_to_base"]["frame_id"]
                args.child_frame_id = data["cam_to_base"]["child_frame_id"]
                args.tx, args.ty, args.tz = data["cam_to_base"]["translation"].values()
                q = data["cam_to_base"]["quaternion"]
                args.qx, args.qy, args.qz, args.qw = q['x'], q['y'], q['z'], q['w']

        self.frame_id = args.frame_id
        self.child_frame_id = args.child_frame_id
        self.tx, self.ty, self.tz = args.tx, args.ty, args.tz
        axis_angle = quaternion_to_axis_angle(args.qw, args.qx, args.qy, args.qz)
        self.qx, self.qy, self.qz = axis_angle
        self.output = args.output

        self.layout = QVBoxLayout()
        self.controls = {}

        for label_name in ["tx", "ty", "tz", "qx", "qy", "qz"]:
            h_layout = QHBoxLayout()
            
            lbl = QLabel(f"<b>{label_name}:</b>")
            lbl.setFixedWidth(30)
            
            spin = QDoubleSpinBox()
            spin.setDecimals(20) 
            
            # Set range: Translations -100 to 100, Quaternions -1 to 1
            if "t" in label_name:
                spin.setRange(-100.0, 100.0)
            else:
                spin.setRange(-100.0, 100.0) # Quaternions stay within -1 and 1
            
            spin.setSingleStep(0.001)
            
            # Get initial value from args
            initial_val = getattr(args, label_name) if hasattr(args, label_name) else 0.0

            setattr(self, label_name, initial_val)
            spin.setValue(initial_val)
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-1000)
            slider.setMaximum(1000)
            slider.setValue(int(initial_val * self.scale))

            spin.valueChanged.connect(lambda val, l=label_name, s=slider: self.update_from_spin(l, val, s))
            slider.valueChanged.connect(lambda val, l=label_name, sp=spin: self.update_from_slider(l, val, sp))

            h_layout.addWidget(lbl)
            h_layout.addWidget(spin)
            h_layout.addWidget(slider)
            self.layout.addLayout(h_layout)
            self.controls[label_name] = spin

        self.save_button = QPushButton("Save Transform")
        self.save_button.clicked.connect(self.save_transform)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)
        self.setWindowTitle("High Precision TF Controller")
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_tf)
        
        threading.Thread(target=self.tf_listener_thread, daemon=True).start()

    def update_from_spin(self, label, value, slider):
        setattr(self, label, value)
        slider.blockSignals(True)
        slider.setValue(int(value * self.scale))
        slider.blockSignals(False)

    def update_from_slider(self, label, value, spin):
        scaled_val = value / self.scale
        setattr(self, label, scaled_val)
        spin.blockSignals(True)
        spin.setValue(scaled_val)
        spin.blockSignals(False)

    def tf_listener_thread(self):
        while not rospy.is_shutdown():
            try:
                self._last_middle_T = transform_to_mat44(
                    self.tf_buffer.lookup_transform(self.world_frame, self.temp_frame, rospy.Time(0))
                )
            except Exception:
                pass
            rospy.sleep(0.05)

    def axis_angle_to_quaternion(self):
        axis = np.array([self.qx, self.qy, self.qz])
        angle = np.linalg.norm(axis)
        if angle > 1e-12: # Increased precision check
            axis_n = axis / angle
            ha = angle / 2.0
            return axis_n[0]*np.sin(ha), axis_n[1]*np.sin(ha), axis_n[2]*np.sin(ha), np.cos(ha)
        return 0.0, 0.0, 0.0, 1.0

    def publish_tf(self, event):
        if self._last_middle_T is None: return
        
        qx, qy, qz, qw = self.axis_angle_to_quaternion()
        T_cam_world = quaternion_matrix([qx, qy, qz, qw])
        T_cam_world[:3, 3] = [self.tx, self.ty, self.tz]

        T_cam_wrist = np.linalg.inv(self._last_middle_T) @ T_cam_world
        q = quaternion_from_matrix(T_cam_wrist)
        
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.child_frame_id
        
        if self.frame_id == self.world_frame:
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = self.tx, self.ty, self.tz
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
        else:
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = T_cam_wrist[:3, 3]
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        
        self.br.sendTransform(t)

    def save_transform(self):
        qx, qy, qz, qw = self.axis_angle_to_quaternion()
        data = {"cam_to_base": {
            "frame_id": self.frame_id, "child_frame_id": self.child_frame_id,
            "quaternion": {"w": float(qw), "x": float(qx), "y": float(qy), "z": float(qz)},
            "translation": {"x": float(self.tx), "y": float(self.ty), "z": float(self.tz)}
        }}
        with open(self.output, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        rospy.loginfo(f"Saved to {self.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str)
    parser.add_argument("--frame_id", type=str, default="world")
    parser.add_argument("--child_frame_id", type=str, default="base_link")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tx", type=float, default=0)
    parser.add_argument("--ty", type=float, default=0)
    parser.add_argument("--tz", type=float, default=0)
    parser.add_argument("--qx", type=float, default=0)
    parser.add_argument("--qy", type=float, default=0)
    parser.add_argument("--qz", type=float, default=0)
    parser.add_argument("--qw", type=float, default=1.0)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    gui = TFGui(args)
    gui.show()
    sys.exit(app.exec_())