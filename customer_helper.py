from groq import Groq
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
import pickle
import re
from dotenv import load_dotenv
load_dotenv()


llama = 'llama3-70b-8192'

model = ChatAnthropic(temperature=0, model_name='claude-3-5-sonnet-20240620')

model_llama = ChatGroq(temperature=0, model_name=llama) 

################################################ Sub question #############################

template_subquestion = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent subquestion generator that extracts subquestions based on human instruction and the CONTEXT provided. You are part of a Text-to-SQL agent.
"""),

    ("human", '''
CONTEXT:
This dataset pertains to Olist, the largest department store on Brazilian marketplaces.
When a customer purchases a product from the Olist store (from a specific seller and location), the seller is notified to fulfill the order.
Once the customer receives the product or the estimated delivery date passes, the customer receives a satisfaction survey via email to rate their purchase experience and leave comments.

You are given:
- A user question
- A list of table names with descriptions

Instructions:
Think like a Text-to-SQL agent. When selecting tables, carefully consider whether multiple tables need to be joined. Only select the tables necessary to answer the user question.
*** A table might not answer a subquestion, but adding it might act as a link with another table selected by different agent. Think in this way while selecting a table. If selected table has all information, ignore other tables.


Your task:
1. Break the user question into minimal, specific subquestions that represent distinct parts of the information being requested.
2. For each subquestion, identify a **single table** whose **description** clearly indicates it contains the needed information.
3. **Ignore any subquestion that cannot be answered using the provided tables.**
4. **Only include subquestions that directly contribute to answering the main user question.**
5. If a subquestion can be answered using multiple tables, intelligently choose the single most appropriate table based on the description.
6. Be highly specific and avoid redundant or irrelevant subquestions. For example, if the number of orders is asked, only use order IDs—no other order details are needed.

Additional Guidelines:
- Fully understand the CONTEXT above before attempting subquestion generation. This is crucial to identifying relevant data.
- You are NOT answering the question itself.
- You are NOT responsible for whether the entire question is answerable from the available data.
- Your ONLY job is to check whether a specific subpart of the question can be answered from a table based on its description.
- If multiple subquestions map to the same table, group them into a single list entry like club multiple subquestions into 1 single question.
- A table might not answer a subquestion, but adding it might act as a link with another table selected by different agent that helps answering user question. Think in this way while selecting a table. If selected table has all information, ignore other tables.
- STRICTLY exclude subquestions that no table can answer.
- Length of each sublist should be exactly 2 as per below output format.

Output format:
Return a list of lists in the following format. Ensure all strings use double quotes. Length of each sublist should be exactly 2. :
[["subquestion1", "table name 1"], ["subquestion2", "table name 2"]]

If multiple subquestions map to the same table:
[["subquestion1", "subquestion2", "table name"]]

If only one valid subquestion:
[["subquestion1", "table name"]]

If no valid subquestions:
[[]]

---

Examples

Question: "Give me the list of customers who have bought more than 5 products in the last month using UPI. Also, list the product categories they purchased."

HOW TO THINK STEP BY STEP:
- Understand the CONTEXT and business process.
- “List of customers” → Check if any table tracks customers → Yes, "customer" table.  Check if any table is needed that has link with customer identifier and helps in answering user question
- “More than 5 products bought” → Check for product identifiers. → Yes, "product" table.  Check if any table is needed that has link with product identifier and helps in answering user question
- “Product categories purchased” → Check for product category info → Yes, "product" table.  Check if any table is needed that has link with product identifier and helps in answering user question
- “Using UPI” → Is there a table with payment method? → No, ignore this subquestion. 
- “Product table” answers two subquestions → Group both.
- Do a final check if i missed anything 

Output:
[
  ["List of customers", "customer"],
  ["Total products bought per customer", "Product categories purchased", "product"]
]

---

Table List:
{tables}

User question:
{user_query}
''')
])



chain_subquestion = (
    RunnableMap({
        "tables": lambda x: x["tables"],
        "user_query": lambda x: x["user_query"]
    })
    | template_subquestion
    | model
    | StrOutputParser()
)

#########################################column selection######################333

template_column = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent data column selector that chooses the most relevant columns from a list of available column descriptions to help answer a subquestion ONLY.
Your selections will be used by a SQL generation agent, so choose **only those columns** that will help write the correct SQL query for a subquestion based on main question.

Act like you're preparing the exact inputs required to build the SQL logic. Also, look at main user question before selecting columns.
BUT main PRIORITY IS TO SELECT columns for subquestion.
"""),

    ("human", '''
     
HOW TO THINK STEP BY STEP:
- For each subquestion mentioned in subquestion below, think if <column1> in Column list might help in answering the question based on column description below. If no, check if this column can be used to answer any part of main question below.
    subquestion, main question: If column1 is used to answer any of these based on column description? If yes, then select that column
    subquestion, main question: If column2 is used to answer any of these based on column description? If yes, then select that column
    and so on.. for all columns

- There can be **critical dependencies between different columns** to answer a subquestion. THIS IS MANDATORY STEP TO THINK LIKE THIS WHEREVER NECESSARY.
  For example, a total value or aggregated metric may require combining:
    - multiple values from repeated rows (e.g., payments, items, events)
    - additional grouping or identifier columns
    - columns that explain how a value is split or repeated (e.g., installments, splits, quantities)

  In such cases, **select all contributing columns together**. Never assume a single column alone is sufficient for aggregated results.

- Include any supporting columns that help define or group the main entity (e.g., order_id if the question asks for order-level info).
- Only after processing the subquestion completely, look at main question to see if it adds any more relevant columns.

RULES:
1. ALWAYS include any **unique identifiers** related to the entity being queried (e.g., order_id, product_id, customer_id).
2. **NEVER select** the `customer_unique_id` column — it must always be ignored.
3. ONLY before generating final list of columns, check main question if any other column helps in answering it.
4. When a metric or value depends on multiple rows, parts, quantities, or repeated fields, you must include all columns required to fully calculate or group that metric.
5. While selecting columns, describe how each selected column **contributes to solving the subquestion**, and include full column description from column list.
6. Output must be in the format specified below. Length of each sublist should be exactly 2.

**Output Format:**
Output should look like below list of lists format for sure.. This is mandatory. Make sure to say there are many other values in that column apart from sample values. Length of each sublist should be exactly 2.
[["<column name 1>", "<Add column description as per column list below + What part of question this column answers something like this column tells about.... sample values:<> and add so on...>"], ["<column name 2>", "<Add column description as per column list below + What part of question this column answers like this column tells about.... .sample values:<> and so on in sample values>"]]

Example: To know total value of an order, order might have multiple items. We might need to consider number of items * price of each item to know cost of order.
Here, to calculate cost of order, we have considered number of items and price of each item. To solve 1 aspect of the question we required both columns.
Carefully look at such scenarios based on column descriptions. Sometimes, there might not be such cases. Be careful and consider descriptions of different column.

---

Column list:
{columns}
     
subquestion:
{query}
     
Main question:
{main_question}
    ''')
])



chain_column_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "main_question": lambda x: x["main_question"]
    })
    | template_column
    | model
    | StrOutputParser()
)


############################### Decision#####################


template_filter_check = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert assistant designed to help a text-to-SQL agent determine whether filters (i.e., WHERE clauses) are required for answering a user's natural language question using a SQL query on a database.

Your job is to:
1. Carefully analyze the user question and identify if any filtering condition is implied (e.g., city = 'Campinas', date range, payment type = 'credit_card', etc.).
2. Use the provided list of tables and columns (with sample values) to identify which specific string datatype **columns** would be involved in such filtering.
     
3. Determine whether a **filter is needed** ONLY for string datatype columns:
    - If **yes**, return a list in the format:
      ["yes", ["<table>", "<column>", "<filter values exactly as stated in the user question>"], ["<table2>", "<column2>", "<filter values exactly as stated in the user question>"], ...]
    - If **no filter is needed**, return: ["no"]
4. For the third item in each filter entry, suggest value(s) **exactly as stated in the user query**, even if they are different from the sample values.
   - If user says "New York" and the column has "SF" it means in actual columns values are in abbrevation, so output "NY". Suggest based on user question and sample values.
   - If user says "credit card or boleto", output "credit card, boleto"
5. Only include columns in the output that help **narrow down** the dataset, such as city, state, payment method etc.
6. For float or integer or DATE datatype columns just give ["no"] as output.  For date kind of columns give output as ["no"]
7. Output should be STRICTLY in form of list.

⚠️ Be careful not to include aggregation or grouping columns like `customer_id`, `order_id`, or `product_id` unless they are being **explicitly** filtered in the question.

Example outputs:
["yes", ["customers", "customer_city", "Campinas"], ["orders", "order_status", "delivered, shipped"]]
["yes", ["order_payments", "payment_type", "credit card, boleto"]]
["no"]
    """),

    ("human", '''
Given a user query and the available list of tables and column names (with sample values), decide if the SQL query to answer this question requires filters.

Only return a list in the exact format described:
- "yes" if filtering is needed, followed by the relevant table-column-filter entries.
- "no" if the question can be answered using full-table aggregates or joins without conditions. For float ot integer or date datatype columns also just give ["no"] as output.
- Make sure that output should be strictly in terms of list or list of lists. Make sure strings within these lists are properly closed by ".

Here is the user question:
{query}

And here is the list of available tables and columns (with sample values):
{columns}
''')
])



chain_filter_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"]
    })
    | template_filter_check
    | model_llama
    | StrOutputParser()
)


########################################## QUERY generation #################################3

template_sql_query = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent MySQL query generator. Your task is to generate syntactically correct and optimized MySQL queries based on the user's question, relevant table/column details, and optional filter values.

You must respect **all the column selections** made by the previous agent. Every selected column is considered essential for logic, traceability, audit, or correctness and **must be used in the query based on its description.**
"""),

    ("human", '''
Instructions:
- You will receive:
  - A user question
  - A list of relevant tables and columns (including column descriptions and sample values) as selected by previous text-to-SQL agent.
  - Optional filter values for specific columns
     
HOW TO THINK STEP BY STEP:
- Carefully read through all columns listed under "Relevant tables and columns". These columns have already been selected after deep reasoning by a specialized agent, so you must treat all of them as **mandatory**.
- Use **all the listed columns** in your query, even if some may not affect the result directly. This is required for traceability and correctness.
- Read the column descriptions carefully. Some columns may influence interpretation, even if they aren’t directly used in filtering or aggregation (e.g., they indicate multiple parts of a transaction, how values are split, or sequencing). Include such columns in the query logic or SELECT clause.
- Do NOT drop or skip any columns under any circumstances.
- Avoid using reserved SQL keywords like or, and, or as as aliases, as they may cause query errors.
- Use CTE if query is big.

- Check if there are any filters mentioned in "Applicable filters" below.
  - If yes, verify that they match column types and values.
  - Apply them using appropriate SQL WHERE clauses.
  - If no filters, proceed to generate the base query.

- Use meaningful table aliases to improve readability in joins. Be very careful while selecting alias. Dont select alias like or, and etc.
- The final query must be:
  - **Avoid using reserved SQL keywords like 'or', 'and', or 'as' as aliases in SQL query, as they may cause query errors**
  - **Syntactically valid**
  - **Optimized for MySQL**
  - **Ready to execute**
  - **Includes all provided columns in logic or SELECT**
  - **If the query involves filtering grouped results or counting grouped records, use subqueries where appropriate to avoid logical conflicts between GROUP BY, HAVING, and aggregate functions in the SELECT clause

User question:
{query}

Relevant tables and columns:
{columns}

Applicable filters:
{filters}
''')
])




chain_query_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "filters": lambda x: x["filters"]
    })
    | template_sql_query
    | model
    | StrOutputParser()
)

############################################# Validation ################
template_validation = ChatPromptTemplate.from_messages([
    ("system", """
You are a highly capable and precise MySQL query validator.

Your role is to:
1. Understand the user's question.
2. Interpret the selected input columns and their descriptions.
3. Remove uncessary columns if any in the output. I need only relavant columns in output. Remove all unecessary columns from select.
4. Analyze the SQL query generated by a prior agent.
5. Validate the query strictly according to syntax, logic, and compliance rules.
6. Remove/REPLACE any reserved SQL keywords like 'or', 'and', or 'as' as aliases, as they may cause query errors in the given SQL
7. If the query involves filtering grouped results or counting grouped records, use subqueries where appropriate to avoid logical conflicts between GROUP BY, HAVING, and aggregate functions in the SELECT clause

Your output should either confirm the existing SQL query (if valid) or provide a corrected version with a clear adherence to best practices.
    """),

    ("human", '''

You must enforce the following rules with **strict accuracy**:

- **All selected columns are mandatory**. Every column provided is crucial for business logic, traceability, auditability, or correctness. **They must appear in the query in a relevant and meaningful way, based on their descriptions.**
- If any selected column is missing, misused, or ignored, the query must be rewritten accordingly.
- If the query involves filtering grouped results or counting grouped records, use subqueries where appropriate to avoid logical conflicts between GROUP BY, HAVING, and aggregate functions in the SELECT clause
- Validate all SQL **aliases** especially in joins—for clarity and consistency.
-  Remove/REPLACE any reserved SQL keywords like 'or', 'and', or 'as' as aliases, as they may cause query errors in the given SQL
- Ensure **complete syntactic correctness** of the SQL statement.
- Ensure logical soundness and alignment with the user’s intent.
- If the provided query is fully correct, return it **unchanged**.
- If it has issues, return a revised and correct version.

---

**User Question:**  
{query}

**Relevant Tables and Columns:**  
{columns}

**Applicable Filters:**  
{filters}

**SQL Query to Validate:**  
{sql_query}
''')
])


chain_query_validator = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "filters": lambda x: x["filters"],
        'sql_query': lambda x: x["sql_query"],
    })
    | template_validation
    | model
    | StrOutputParser()
)
