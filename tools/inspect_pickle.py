import pickle
import sys
import os

filename = "best_model_detection.pkl"

if not os.path.exists(filename):
    print(f"File {filename} not found in current directory.")
    sys.exit(1)

print(f"Loading {filename}...")
try:
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    print("Keys in top-level dictionary:")
    print(list(data.keys()))

    if 'params' in data:
        print(f"'params' found with type: {type(data['params'])}")
        # Check if params is a dict or something else.
        if isinstance(data['params'], dict):
             print(f"  params keys (first 5): {list(data['params'].keys())[:5]}")
        
    if 'batch_stats' in data:
        print(f"'batch_stats' found with type: {type(data['batch_stats'])}")
        if isinstance(data['batch_stats'], dict):
             # Try to see if it's empty
             if not data['batch_stats']:
                 print("  batch_stats is EMPTY!")
             else:
                 print(f"  batch_stats keys (first 5): {list(data['batch_stats'].keys())[:5]}")
    else:
        print("'batch_stats' NOT found in top-level dict.")

    if 'model_state' in data:
         print(f"'model_state' found with type: {type(data['model_state'])}")
         model_state = data['model_state']
         if isinstance(model_state, dict):
            print(f"  model_state keys: {list(model_state.keys())}")
            if 'batch_stats' in model_state:
                print(f"  'batch_stats' found in model_state with type: {type(model_state['batch_stats'])}")
                if not model_state['batch_stats']:
                     print("    model_state['batch_stats'] is EMPTY!")
                else:
                     print(f"    model_state['batch_stats'] keys: {list(model_state['batch_stats'].keys())[:5]}")
            else:
                print("  'batch_stats' NOT found in model_state.")

except Exception as e:
    print(f"Error loading pickle: {e}")
