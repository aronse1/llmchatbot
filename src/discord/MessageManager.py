import redis
import json
import time

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

async def save_message(user_id, role, message):
    history = get_history(user_id)
    history.append({'role': role, 'message': message, 'timestamp': time.time()})
    r.set(user_id, json.dumps(history), ex=600)  

async def get_history(user_id):
    history = r.get(user_id)
    return json.loads(history) if history else []

async def clear_history(user_id):
    r.delete(user_id)
