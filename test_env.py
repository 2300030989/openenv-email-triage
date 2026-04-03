from email_env import EmailEnv
from schema import Action, ActionType

def test_easy_task():
    env = EmailEnv(task_id="easy")
    obs = env.reset()
    assert len(obs.inbox) == 6 # 5 newsletters + 1 mom email
    
    # Archive 5 newsletters
    for i in range(5):
        action = Action(action_type=ActionType.ARCHIVE, email_id=f"news_{i}")
        obs, reward, done, info = env.step(action)
    
    assert env.grade() == 1.0
    assert done == True
    print("Easy task test passed!")

def test_medium_task():
    env = EmailEnv(task_id="medium")
    obs = env.reset()
    
    # Reply to boss
    action_reply = Action(action_type=ActionType.REPLY, email_id="boss_1", content="I'll be there!")
    env.step(action_reply)
    
    # Create calendar event
    action_cal = Action(action_type=ActionType.CREATE_CALENDAR_EVENT, event_details={"title": "Review", "time": "Friday 10am"})
    obs, reward, done, info = env.step(action_cal)
    
    assert env.grade() == 1.0
    assert done == True
    print("Medium task test passed!")

if __name__ == "__main__":
    test_easy_task()
    test_medium_task()
