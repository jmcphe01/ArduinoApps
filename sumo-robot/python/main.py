import sys
print(sys.version)   # printed first, so you see it even if the import fails

import time
import legoeducation as le

motor = le.SingleMotor()
motor.connect(card_color=le.LEGO_COLOR_ORANGE, card_serial="1129")

if not motor.connected:
    print("Error connecting to Single Motor.")
    exit(1)

print("Position:", motor.motor.position)
motor.motor_run_for_degrees(45)
time.sleep(1)
print("Position:", motor.motor.position)
motor.motor_run_for_degrees(45, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
motor.disconnect()