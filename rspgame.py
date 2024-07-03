from flask import Flask, render_template, jsonify, request
import numpy as np
import base64
import cv2
import random

app = Flask(__name__)

# Dummy class names for demonstration
class_names = ['paper', 'rock', 'scissors']

# Dummy function to simulate game logic
def play_game(image):
    # Simulate computer's choice
    computer_choice = random.choice(class_names)

    # Dummy logic for player's choice (replace with actual prediction logic)
    # For demonstration, assume player's choice is based on image classification
    player_choice = class_names[np.random.randint(0, len(class_names))]

    # Dummy result logic (replace with actual game logic)
    if computer_choice == player_choice:
        result = '비김'
    elif (computer_choice == 'scissors' and player_choice == 'rock') or \
         (computer_choice == 'rock' and player_choice == 'paper') or \
         (computer_choice == 'paper' and player_choice == 'scissors'):
        result = '플레이어가 이김'
    else:
        result = '컴퓨터가 이김'

    return computer_choice, player_choice, result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play_game', methods=['POST'])
def play_game_route():
    try:
        # Get base64 encoded image from request
        frame_data = request.json['image']
        # Decode base64 image data
        nparr = np.frombuffer(base64.b64decode(frame_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Call function to play game and get result
        computer_choice, player_choice, result = play_game(frame)

        # Prepare response JSON
        response = {
            'computer': computer_choice,
            'player': player_choice,
            'result': result
        }

        return jsonify(response)

    except Exception as e:
        print("Error during game play:", e)
        return jsonify({'error': 'Error during game play'})

if __name__ == '__main__':
    app.run(debug=True)
