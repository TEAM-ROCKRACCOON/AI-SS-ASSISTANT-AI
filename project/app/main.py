from datetime import date, timedelta
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path
from pprint import pprint

from experiment import convert_to_weeklyTodos, run_one_week 

app = FastAPI(title="AI-SS API", version="1.0.0")

# JSON 파일 경로 설정
data_dir = Path(__file__).resolve().parent.parent / "data"
base_date_path = data_dir / "base_date.json"
user_cleaning_status_path = data_dir / "user_cleaning_status.json"
behavior_history_path = data_dir / "behavior_history.json"
weekly_todo_history_path = data_dir / "weekly_todo_history.json"

class OneWeekInput(BaseModel):
    week_start: date
    seed: int = 42

def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}

base_date = load_json(base_date_path)
user_cleaning_status = load_json(user_cleaning_status_path)
behavior_history = load_json(behavior_history_path)
weekly_todo_history = load_json(weekly_todo_history_path)

# -------- Endpoint -------- 
@app.post("/api/v1/schedule/run-one-week") 
def generate_one_week_schedule(body: OneWeekInput):
    global base_date  
 
    if not base_date:
        base_date["date"] = str(body.week_start)

    start_date = date.fromisoformat(base_date["date"])
    delta_days = (body.week_start - start_date).days
    this_week = (delta_days // 7) + 1

    prev_behavior = behavior_history.get(str(body.week_start - timedelta(days=7)), None)
    last_week_todo = weekly_todo_history.get(str(body.week_start - timedelta(days=7)), None)

    behavior_vector, weekly_todo = run_one_week(
        this_week, body.week_start, last_week_todo, prev_behavior, user_cleaning_status
    )
 
    behavior_history[str(body.week_start)] = (
        behavior_vector.tolist() if isinstance(behavior_vector, np.ndarray) else behavior_vector
    ) 
    weekly_todo_history[str(body.week_start)] = weekly_todo

    # JSON 파일 저장
    with open(behavior_history_path, "w", encoding="utf-8") as f:
        json.dump(behavior_history, f, ensure_ascii=False, indent=4)

    with open(weekly_todo_history_path, "w", encoding="utf-8") as f:
        json.dump(weekly_todo_history, f, ensure_ascii=False, indent=4)

    with open(base_date_path, "w", encoding="utf-8") as f:
        json.dump(base_date, f, ensure_ascii=False, indent=4)

    weeklyTodos = convert_to_weeklyTodos(weekly_todo, body.week_start)

    return {"weeklyTodos": weeklyTodos}


if __name__ == "__main__":
    
    # # 테스트 예시
    # for week_str in ["2025-11-03", "2025-11-10", "2025-11-17"]:    
    #     body = OneWeekInput(
    #         week_start=week_str
    #     )
    #     resp = generate_one_week_schedule(body) 
    #     pprint(resp)
    
    pass