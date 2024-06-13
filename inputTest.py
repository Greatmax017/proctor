import pyaudio

p = pyaudio.PyAudio()

# Get the available input devices
input_devices = [p.get_device_info_by_index(i) for i in range(p.get_host_api_info_by_index(0).get('deviceCount'))]

# Print the device names
for i, device in enumerate(input_devices):
    if device.get('maxInputChannels') > 0:
        print(f"{i}: {device.get('name')}")

# Select the desired input device index
device_index = int(input("Enter the desired input device index: "))

# Open the stream with the selected device
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_index)