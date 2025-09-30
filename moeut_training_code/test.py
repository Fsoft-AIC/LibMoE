import subprocess

while True:
    result = subprocess.run(["python", "/cm/shared/anonymous/moeut_training_code/main.py"],
                            capture_output=True, text=True)
    print("Subprocess Output:", result.stdout)
    if "@@@@@" in result.stdout:
        break
