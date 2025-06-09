import pandas as pd
from sqlalchemy import create_engine,  text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import Integer, Float, String
from rapidfuzz import process, fuzz

engine = create_engine('mysql+mysqlconnector://root:Indianarmy@localhost/txt2sql')


def get_best_fuzzy_match(input_value, choices):

    match, score, _ = process.extractOne(input_value, choices, scorer=fuzz.token_set_ratio)
    return match, score


def get_values(table_name, column_name):

    # SQL query to get distinct values
    query = f"SELECT DISTINCT {column_name} FROM {table_name}"

    # Execute query and load results into DataFrame
    df = pd.read_sql(query, con=engine)

    # Optionally, convert to list if you want raw values
    unique_values = df[column_name].dropna().tolist()

    return unique_values


def call_match(val):
    final = []
    for lst in val[1:]:
        table = lst[0]
        column = lst[1]
        str_lst = [i.strip() for i in lst[2].split(',')]

        unq_col_val = get_values(table, column)
        unq_col_val = [str(i) for i in unq_col_val]

        for subval in str_lst:
            best_match, score = get_best_fuzzy_match(subval, unq_col_val)
            final.append(["table name:"+table, "column_name:"+column, "filter_value:"+best_match])

    return final