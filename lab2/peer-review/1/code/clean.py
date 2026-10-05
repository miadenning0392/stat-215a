from pathlib import Path
import pandas as pd
from pyreadr import read_r

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_raw(data_dir=DATA_DIR):
    '''Read lingData, lingLocation and the question text'''
    ling_data = pd.read_csv(data_dir / "lingData.txt", sep=r"\s+", dtype={"ZIP": str})
    ling_location = pd.read_csv(data_dir / "lingLocation.txt", sep=r"\s+")
    question_data = read_r(data_dir / "question_data.RData")
    return ling_data, ling_location, question_data

def load_questions(question_data):
    '''Return a dict of questions and their answers'''
    questions = {}
    for qnum, quest in zip(question_data["quest.use"]["qnum"], question_data["quest.use"]["quest"]):
        num = int(qnum)
        answers = list(question_data[f"ans.{num}"].sort_values("ans.let")["ans"].str.strip())
        questions[f"Q{num:03d}"] = {"question": quest, "answers": answers}
    return questions

def answer_text(questions, question, code):
    '''Answer text for an answer code, e.g. answer_text(questions, "Q105", 2) -> "pop"'''
    return questions[question]["answers"][int(code) - 1]

def clean_data(df, max_nonresponse=66):
    qcols = [col for col in df.columns if col.startswith('Q')]
    df = df.dropna(subset=['lat', 'long'])
    n_missing = (df[qcols]==0).sum(axis=1)
    df = df[n_missing <= max_nonresponse]

    return df.reset_index(drop=True)

def onehot_encode(ling_data, questions):
    '''One-hot encode the responses. Non-responses become zero'''
    qcols = [c for c in ling_data.columns if c.startswith("Q")]
    answers = pd.DataFrame({
        # turn 0 (no answer) into NaN first, so it gets all zeros
        q: pd.Categorical(ling_data[q].where(ling_data[q] > 0),
                          categories=range(1, len(questions[q]["answers"]) + 1))
        for q in qcols
    })
    return pd.get_dummies(answers).astype(float)

def load_clean(data_dir=DATA_DIR, **clean_options):
    '''Run everything. Returns clean_df, X (one-hot), questions'''
    ling_data, ling_location, question_data = load_raw(data_dir)
    questions = load_questions(question_data)
    clean_df = clean_data(ling_data, **clean_options)
    X = onehot_encode(clean_df, questions)
    return clean_df, X, questions
