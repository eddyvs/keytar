import pyaudio
import numpy as np
import aubio
import librosa
import pydirectinput
from cmu_graphics import *
import threading
from collections import deque
import time

############################## PYAUDIO CONFIG ###########################

p = pyaudio.PyAudio()

CHUNK = 1024
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_DEVICE_INDEX = 1

try:
    device = p.get_device_info_by_index(INPUT_DEVICE_INDEX)
except Exception:
    raise Exception("no microphone connected")

audioStream = p.open(
        input=True,
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=CHUNK,
    )

########################## AUBIO PITCH DETECTION #######################

pitch_detector = aubio.pitch("yin", 2048, CHUNK, RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_tolerance(0.8)

############ PYDIRECTINPUT CONFIG ################

pydirectinput.PAUSE = 0.01

################## MORSE CODE CONFIG ####################################

MORSE_DOT_NOTE = "C3"
MORSE_DASH_NOTE = "C♯3"
TOGGLE_NOTE = "F♯3"
MORSE_LETTER_GAP = 1.0   

MORSE_CODES = {
    ".-": "a", "-...": "b", "-.-.": "c", "-..": "d",
    ".": "e", "..-.": "f", "--.": "g", "....": "h",
    "..": "i", ".---": "j", "-.-": "k", ".-..": "l",
    "--": "m", "-.": "n", "---": "o", ".--.": "p",
    "--.-": "q", ".-.": "r", "...": "s", "-": "t",
    "..-": "u", "...-": "v", ".--": "w", "-..-": "x",
    "-.--": "y", "--..": "z",
}

##########################           KEYBINDS               ############################

keyBinds = {
"D3":"a",
"D♯3": "s",
"E3": "d",   
"G♯3": "w", 
"F3": "up",
"C3": "down",
"C♯3": "right",
"B2": "left",
"A♯3": "click",
"B3": "rightclick",
"A3": "space",
"F♯3": "togglemorse"
}



note_queue = deque(maxlen=3)

def keybindLoop():
    previousKeybind = None
    previousNote = None
    morseMode = False
    morseSymbols = []
    silenceStarted = None
    lastDecodedLetter = ""


    while True:
        frequency = getFrequency()
        noteInRange = frequency >= 90 and frequency <= 1200 and pitch_detector.get_confidence()>0.6

        if noteInRange==False:
            if previousNote is not None:
                previousNote = None
                if morseMode:
                    now = time.monotonic()
                    silenceStarted = now
                else:
                    keyUpAll()
                    previousKeybind = None
                enqueDisplayUpdate(None, None, morseMode, morseSymbols,
                                  lastDecodedLetter)

            if (morseMode and morseSymbols and silenceStarted is not None
                    and now - silenceStarted >= MORSE_LETTER_GAP):
                lastDecodedLetter = finishMorseLetter(morseSymbols)
                morseSymbols.clear()
                silenceStarted = None
                enqueDisplayUpdate(None, None, morseMode, morseSymbols,
                                  lastDecodedLetter)
            continue

        note = librosa.hz_to_note(frequency)
        noteChanged = note != previousNote
        previousNote = note

        if note == TOGGLE_NOTE:
            if noteChanged:
                morseMode = not morseMode
                morseSymbols.clear()
                lastDecodedLetter = ""
                keyUpAll()
                previousKeybind = None
                enqueDisplayUpdate(note, "togglemorse", morseMode,
                                  morseSymbols, lastDecodedLetter)
            continue

        if morseMode:
            if noteChanged:
                if note == MORSE_DOT_NOTE:
                    morseSymbols.append(".")
                    keybind = "DOT"
                elif note == MORSE_DASH_NOTE:
                    morseSymbols.append("-")
                    keybind = "DASH"
                else:
                    keybind = "ignored"
                enqueDisplayUpdate(note, keybind, morseMode, morseSymbols,
                                  lastDecodedLetter)
            continue

        keyBind = keyBinds.get(note)

        if keyBind is None:
            keyUpAll()
            previousKeybind = None
            if noteChanged:
                enqueDisplayUpdate(note, "unmapped", morseMode, morseSymbols,
                                  lastDecodedLetter)
            continue

        handleInput(keyBind, previousKeybind)
        
        previousKeybind = keyBind
        if noteChanged:
            enqueDisplayUpdate(note, keyBind, morseMode, morseSymbols,
                              lastDecodedLetter)

def enqueDisplayUpdate(note, keybind, morseMode, morseSymbols, lastDecodedLetter):
    note_queue.append((note, keybind, morseMode,
                       "".join(morseSymbols), lastDecodedLetter))

def handleInput(keyBind, previousKeybind):

    if keyBind == "click" and keyBind==previousKeybind:
        pydirectinput.click()
    elif keyBind== "rightclick" and keyBind == previousKeybind:
        pydirectinput.rightClick()
    elif keyBind in ["up", "down", "left", "right", "space"] and keyBind == previousKeybind:
        
        if keyBind=="up":
            pydirectinput.move(0,-10)
        elif keyBind=="down":
            pydirectinput.move(0,10)
        elif keyBind=="left":
            pydirectinput.move(-10,0)
        elif keyBind=="right":
            pydirectinput.move(10,0)
        elif keyBind == "space":
            pydirectinput.press("space")
            
    elif keyBind == previousKeybind:
        pydirectinput.keyDown(keyBind)
    else:
        keyUpAll()

def keyUpAll():
    keyBoardKeys = ["w", "a", "s", "d", "space"]

    for key in keyBoardKeys:
        try:
            pydirectinput.keyUp(key)
        except Exception:
            pass

def getFrequency():
    frame = audioStream.read(CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    frequency = float(pitch_detector(samples)[0])
    return frequency 

def finishMorseLetter(morseSymbols):
    letter = MORSE_CODES.get("".join(morseSymbols))
    if letter is None:
        return "?"
    pydirectinput.press(letter)
    return letter

################### CMU GRAPHICS #################

def redrawAll(app):
    drawLabel(f"Device: {app.device.get('name')}", 140, 10, size = 14)
    drawLabel("Computer keybinds", 400, 45, size=20, bold=True)
    drawLabel("Play F♯3 to toggle morse-mode!", 400, 125)
    drawLabel(f"Mode: {'Morse' if app.morse_mode else 'Normal'}", 400, 75, size=14)
    drawLabel(f"Note: {app.note}    Key: {app.keybind or '--'}", 400, 105, size=14)
    
    drawComputerKeyboard(app)
    drawSquares(app)

    if app.morse_mode == True:
        drawLabel(f"Morse: {app.morse_code or '--'}    Last letter: {app.decoded_letter or '--'}",
                  400, 390, size=16)
        drawLabel("C3 = dot    D3 = dash    1-second pause sends letter",
                  400, 420, size=13)
        drawLabel("F♯3 toggles Morse mode", 400, 445, size=13)


def onStep(app):

        try:
            note, keybind, morseMode, morseCode, decodedLetter = note_queue.popleft()
        except IndexError:
            return None
        app.morse_mode = morseMode
        app.morse_code = morseCode


        app.decoded_letter = decodedLetter
        if note is None:
            app.note = "Listening..."
            app.keybind = ""
        elif keybind is None:
            app.note = note
            app.keybind = "unmapped"
        else:
            app.note = note
            app.keybind = keybind

            if app.keyColors.get(keybind) != None:
                app.squares.append((app.squareX, app.squareY, 20, 20, app.keyColors.get(keybind)))
                app.squareX+=20

            if app.squareX == 800:
                app.squareX = 0
                app.squareY+=20

def onAppStart(app):
    app.squares = []
    app.squareX = 0
    app.squareY = 340
    app.note = "Waiting for note"
    app.keybind = ""
    app.morse_mode = False
    app.morse_code = ""
    app.decoded_letter = ""
    app.device = device
    app.keyColors = {
"a": "red",
"s": "orange",
"d": "yellow",   
"w": "green", 
"up": "blue",
"down": "purple",
"right": "black",
"left": "pink",
"click": "brown",
"rightclick": "grey",
"space": "teal",
    }

def main():
    runApp(width=800, height=600)

def drawComputerKey(app, keyLabel, keybind, keyLeft, keyTop, keyWidth=50):
    keyHeight = 56

    isPressed = app.keybind == keybind
    fill = "lightGreen" if isPressed else "white"
    drawRect(keyLeft, keyTop, keyWidth, keyHeight, fill=fill, border="black")
    drawLabel(keyLabel, keyLeft + keyWidth / 2, keyTop + 28, bold=True, size=14)
    
    if isPressed:
        drawLabel(app.note, keyLeft + keyWidth / 2, keyTop + 42, size=9)

def drawSquares(app):
    for square in app.squares:
        x, y, width, height, fill = square
        drawRect(x, y, width, height, fill=fill)

def drawComputerKeyboard(app):
    drawComputerKey(app, "W", "w", 131, 155)
    drawComputerKey(app, "A", "a", 76, 215)
    drawComputerKey(app, "S", "s", 131, 215)
    drawComputerKey(app, "D", "d", 186, 215)
    drawComputerKey(app, "SPACE", "space", 76, 275, 517)

    drawComputerKey(app, "UP", "up", 302, 155)
    drawComputerKey(app, "LEFT", "left", 249, 215)
    drawComputerKey(app, "DOWN", "down", 302, 215)
    drawComputerKey(app, "RIGHT", "right", 355, 215)

    drawComputerKey(app, "CLICK", "click", 421, 215, 60)
    drawComputerKey(app, "RIGHTCLICK", "rightclick", 485, 215, 108)

keybindThread = threading.Thread(target=keybindLoop, daemon=True)

keybindThread.start()
main()

keybindThread.join()
