import os
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith('.py'):
            print file
