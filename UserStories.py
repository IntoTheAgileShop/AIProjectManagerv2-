import warnings

warnings.filterwarnings('ignore')
import streamlit as st
from crewai import Agent, Crew, Task, Process
import os
from packaging.version import parse

parse("3.9")

# Streamlit UI for file upload
st.title("User Requirements Writing Assistant")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("Business_Requirements.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Set Environment Variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

    # Step 2: Define Tools
    from crewai_tools import PDFSearchTool

    # Specify the file path relative to the current working directory
    pdf_tool = PDFSearchTool(pdf='Business_Requirements.pdf')

    # Agent 1: PO
    PO = Agent(
        role="Product Owner",
        goal="Transform business requirements in user story format",
        tools=[pdf_tool],
        verbose=True,
        backstory=
        ("As an Agile Product Owner, you transform business requirements from stakeholders into user story format."
         ),
    )

    # Task for PO Agent: User Story writing
    userstory_writing = Task(
        description=
        ("Analyze the requirements file provided and transform the business requirements into user story format."
         ),
        expected_output=
        ("A structured list of user stories. Example format: "
         "As a LinkedIn user, I want to search for jobs that are remote only, "
         "so that I can apply to jobs that allow me to work from any location."
         ),
        agent=PO,
        tools=[pdf_tool],
    )

    # Agent 2: BA
    BA = Agent(
        role="Technical Business Analyst",
        goal=
        "Complete user stories with acceptance criteria and add Give, When, Then format",
        tools=[],
        verbose=True,
        backstory=
        ("After the Product Owner writes the basic user story format. Your job is to add acceptance Criteria in the format Given When Format."
         ),
    )

    # Task for BA Agent: Write Acceptance Criteria
    ac_writing = Task(
        description=
        ("Using the user stories obtained from the other task, add an acceptance criteria for each user story in the format Given When Then"
         ),
        expected_output=
        ("A structured list of user stories that includes acceptance criteria. Example format: Given I am under the job tab in LinkedIn When I search for jobs And I filter by remote only Then remote only jobs are displayed And I can apply for them"
         ),
        agent=BA,
        dependencies=[userstory_writing],
    )

    # Agent 3: QA
    QA = Agent(
        role="Quality Assurance",
        goal=
        "Validate that requirements are transformed in user story format with acceptance criteria",
        tools=[],
        verbose=True,
        backstory=
        ("You are a quality assurance that verifies the user stories and the requirements from the BA agent. Validated the user story are in user story format and have acceptance criteria in the format of Give When Then format."
         ),
    )

    # Task for QA Agent: Validate user story format
    qa_validation = Task(
        description=
        ("Using the user stories with acceptance criteria obtained from the AC task from the BA agent, validate each user story has a user story format with acceptance criteria in give, when, then, format."
         ),
        expected_output=
        ("Each requirement has been transformed in a user story format with acceptance criteria in given when format.They should be all be listed in a structural easy to read way."
         ),
        agent=QA,
        dependencies=[ac_writing],
    )

    # Form the crew
    crew = Crew(
        agents=[PO, BA, QA],
        tasks=[userstory_writing, ac_writing, qa_validation],
        process=Process.sequential,
        verbose=2,
    )

    # Kickoff the process
    result = crew.kickoff()

    # Display the result
    st.subheader("Result of CrewAI Process")
    st.write(result)

else:
    st.warning("Please upload a PDF file to proceed.")
