
from pathlib import Path

def build_solve_task_prompt(task:str)->str:
    return f"""
    {task}
    """
    
def build_judge_task_prompt(task:str, conversation_history:str)->str:
    template_path = Path(__file__).with_name("response_quality_check.md")
    template = template_path.read_text(encoding="utf-8")
    return (
        template
        .replace("{QUESTION_CONTENT}", task)
        .replace("{CONVERSATION_HISTORY}", conversation_history)
    )