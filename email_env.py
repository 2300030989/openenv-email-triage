import random
from typing import List, Tuple, Dict, Any, Optional
from base import OpenEnv
from schema import Observation, Action, Reward, Email, ActionType, Action as PydanticAction
from datetime import datetime

class EmailEnv(OpenEnv):
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.max_steps = 20
        self.current_step = 0
        self.inbox: List[Email] = []
        self.calendar: List[Dict[str, Any]] = []
        self.archived_ids: List[str] = []
        self.replied_ids: List[str] = []
        self.forwarded_ids: List[str] = []
        self.urgent_ids: List[str] = []
        self.reset()

    def _generate_emails(self):
        emails = []
        if self.task_id == "easy":
            # Task: Archive 5 newsletters
            for i in range(5):
                emails.append(Email(
                    id=f"news_{i}",
                    sender="newsletter@weekly.com",
                    subject=f"Weekly Update #{i}",
                    body="Here is your weekly summary of things you don't need to read.",
                    timestamp="2024-03-20 09:00",
                    is_read=False
                ))
            emails.append(Email(
                id="important_1",
                sender="mom@home.com",
                subject="Dinner tonight?",
                body="Are you coming over for dinner at 7?",
                timestamp="2024-03-20 10:00",
                is_read=False
            ))
        
        elif self.task_id == "medium":
            # Task: Reply to boss and schedule meeting
            emails.append(Email(
                id="boss_1",
                sender="boss@company.com",
                subject="Quarterly Review",
                body="Hi, let's meet Friday at 10 AM to discuss the review. Please confirm and add it to the calendar.",
                timestamp="2024-03-20 08:30",
                is_read=False
            ))
            emails.append(Email(
                id="news_1",
                sender="tech@crunch.com",
                subject="Breaking Tech News",
                body="AI is still happening.",
                timestamp="2024-03-20 08:45",
                is_read=False
            ))

        elif self.task_id == "hard":
            # Task: Support escalation
            emails.append(Email(
                id="cust_1",
                sender="angry_customer@client.com",
                subject="URGENT: PRODUCTION DOWN",
                body="The system is completely unresponsive. We are losing $10k per hour. HELP!",
                timestamp="2024-03-20 11:15",
                is_read=False,
                is_urgent=True
            ))
            emails.append(Email(
                id="news_2",
                sender="spam@offers.com",
                subject="Cheap Flights!",
                body="Buy now!",
                timestamp="2024-03-20 11:20",
                is_read=False
            ))
            
        return emails

    def reset(self, seed: int = None) -> Observation:
        if seed is not None:
            random.seed(seed)
        
        self.current_step = 0
        self.inbox = self._generate_emails()
        self.calendar = []
        self.archived_ids = []
        self.replied_ids = []
        self.forwarded_ids = []
        self.urgent_ids = []
        
        return self.state()

    def state(self) -> Observation:
        return Observation(
            inbox=[e for e in self.inbox if e.id not in self.archived_ids],
            unread_count=len([e for e in self.inbox if not e.is_read and e.id not in self.archived_ids]),
            calendar_events=self.calendar,
            last_action_status="ready"
        )

    def step(self, action: PydanticAction) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.current_step += 1
        reward_val = 0.0
        reason = "Action processed."
        partial_progress = 0.0
        
        # Handle Actions
        if action.action_type == ActionType.ARCHIVE:
            if any(e.id == action.email_id for e in self.inbox):
                self.archived_ids.append(action.email_id)
                reward_val += 0.5
                reason = f"Archived {action.email_id}"
            else:
                reward_val -= 1.0
                reason = "Attempted to archive non-existent email."

        elif action.action_type == ActionType.REPLY:
            self.replied_ids.append(action.email_id)
            reward_val += 1.0
            reason = f"Replied to {action.email_id}"

        elif action.action_type == ActionType.FORWARD:
            self.forwarded_ids.append(action.email_id)
            reward_val += 1.0
            reason = f"Forwarded {action.email_id} to {action.recipient}"

        elif action.action_type == ActionType.CREATE_CALENDAR_EVENT:
            self.calendar.append(action.event_details)
            reward_val += 2.0
            reason = "Created calendar event."

        elif action.action_type == ActionType.MARK_URGENT:
            self.urgent_ids.append(action.email_id)
            reward_val += 0.5
            reason = f"Marked {action.email_id} as urgent."

        # Task Specific Grading & Termination
        done = self.current_step >= self.max_steps
        score = self.grade()
        
        if score >= 1.0:
            done = True
            reward_val += 5.0 # Completion bonus
            reason = "Task completed successfully!"
            partial_progress = 1.0
        else:
            partial_progress = score

        return self.state(), Reward(value=reward_val, reason=reason, partial_progress=partial_progress), done, {"score": score}

    def grade(self) -> float:
        """Programmatic grader for the current task."""
        if self.task_id == "easy":
            # Grade: How many newsletters archived out of 5
            newsletters = [f"news_{i}" for i in range(5)]
            archived_news = [id for id in self.archived_ids if id in newsletters]
            return min(1.0, len(archived_news) / 5.0)

        elif self.task_id == "medium":
            # Grade: Replied to boss AND calendar event exists
            replied = "boss_1" in self.replied_ids
            cal_exists = len(self.calendar) > 0
            return (0.5 if replied else 0.0) + (0.5 if cal_exists else 0.0)

        elif self.task_id == "hard":
            # Grade: Urgent marked, Forwarded to eng, Replied to customer
            is_marked = "cust_1" in self.urgent_ids
            is_forwarded = "cust_1" in self.forwarded_ids
            is_replied = "cust_1" in self.replied_ids
            return (0.33 if is_marked else 0.0) + (0.33 if is_forwarded else 0.0) + (0.34 if is_replied else 0.0)
        
        return 0.0

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "EmailTriageEnv",
            "version": "1.0.0",
            "tasks": ["easy", "medium", "hard"],
            "description": "Simulate a busy inbox where an agent must triage emails and manage a calendar."
        }
