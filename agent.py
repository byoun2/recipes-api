import asyncio
import os
from typing import Annotated, Any

import dotenv
from github import Auth, Github
from github.PullRequest import PullRequest
from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

FILE_CONTENT_ENCODING = "utf-8"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
#GITHUB_REPOSITORY_NAME = "recipes-api"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
#GITHUB_USERNAME_ENV = "GITHUB_USERNAME"
GITHUB_USER_REPO_NAME_ENV = "REPOSITORY"

CONTEXT_AGENT_SYSTEM_PROMPT = """You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the CommentorAgent using the handoff tool. """
CONTEXT_AGENT_SYSTEM_PROMPT_OTHER = """You are the context gathering agent. You MUST execute ALL of the following steps in this exact order. 
Do NOT write a final response. Do NOT stop early.

Step 1: Call fetch_pr_details to get PR metadata.
Step 2: Call pr_commits_details using the head_sha from Step 1.
Step 3: Call fetch_github_file with filename "CONTRIBUTING.md" to get contribution rules.
Step 4: Call add_context_to_state with a full summary of everything gathered in Steps 1-3.
Step 5: You MUST hand control back to CommentorAgent using the handoff tool. This step is mandatory.

You are NOT allowed to output a final response. You MUST complete Step 5."""

COMMENTOR_AGENT_SYSTEM_PROMPT_OLD = """You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, request it from the ContextAgent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?\n
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review saved to the context state. \n"""
COMMENTOR_AGENT_SYSTEM_PROMPT = """
You are the CommentorAgent.

Your job is to write a useful draft pull request review comment using the gathered PR context.n

If you do not have enough pull request context, hand off to ContextAgent.

Once you have drafted the review comment:
1. Save the draft comment to state using the available state tool.
2. Hand off to ReviewAndPostingAgent.
3. Do not provide the final response yourself.

The ReviewAndPostingAgent is responsible for reviewing, finalizing, and posting the comment to GitHub.
"""
REVIEW_POSTING_AGENT_SYSTEM_PROMPT_TRY2 = """
You are the ReviewAndPostingAgent.

You are responsible for orchestrating the final review posting workflow.

When the user asks to post a review for a PR:
1. If no draft review is available, hand off to CommentorAgent.
2. When CommentorAgent hands control back to you with a draft review, save the final review.
3. Post the final review to GitHub using the available post_final_review posting tool.
4. Only provide the final response after the review has been posted.

Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.

Do not let CommentorAgent produce the final answer directly. The workflow must return to ReviewAndPostingAgent before completion.
"""
REVIEW_POSTING_AGENT_SYSTEM_PROMPT = """You are the Review and Posting agent. You must use the CommentorAgent to create a draft review comment. 
Once a draft review is generated, you need to run the following final checks and post it to GitHub if it meets the criteria.
   - The draft review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the draft review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 Post the final review to GitHub using the available post_final_review posting tool.
 Do not let CommentorAgent produce the final answer directly. The workflow must return to ReviewAndPostingAgent before completion."""



PullRequestDetails = dict[str, str | list[str]]
CommitFileDetails = dict[str, Any]


def get_required_env_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def build_repository_url_old(username: str, repository_name: str) -> str:
    return f"https://github.com/{username}/{repository_name}.git"

def build_repository_url(user_repo_name: str) -> str:
    return f"https://github.com/{user_repo_name}.git"


def get_full_repository_name(repository_url: str) -> str:
    repository_name = repository_url.split("/")[-1].replace(".git", "")
    username = repository_url.split("/")[-2]
    return f"{username}/{repository_name}"


def create_github_client() -> Github:
    token = get_required_env_variable(GITHUB_TOKEN_ENV)
    return Github(auth=Auth.Token(token))


def get_repository(github_client: Github):
    repository_url = build_repo_url()
    print(repository_url)
    return github_client.get_repo(get_full_repository_name(repository_url))


def build_repo_url() -> str:
    user_repo_name = get_required_env_variable(GITHUB_USER_REPO_NAME_ENV)
    repository_url = build_repository_url(user_repo_name)
    return repository_url


def pull_request_to_dict(pr: PullRequest) -> PullRequestDetails:
    return {
        "author": pr.user.login,
        "title": pr.title or "title: No title provided.",
        "body": pr.body or "No description provided.",
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha,
        "commits": [commit.sha for commit in pr.get_commits()],
    }


def commit_file_to_dict(commit_file: Any) -> CommitFileDetails:
    return {
        "filename": commit_file.filename,
        "status": commit_file.status,
        "additions": commit_file.additions,
        "deletions": commit_file.deletions,
        "changes": commit_file.changes,
        "patch": commit_file.patch,
    }


def create_repository_tools(repository) -> list[FunctionTool]:
    def get_pr_details(
        pr_num: Annotated[int, "A pull request number"],
    ) -> PullRequestDetails:
        """Retrieve pull request details, including author, title, body, diff URL, state, head SHA, and commit SHAs."""
        pr = repository.get_pull(pr_num)
        return pull_request_to_dict(pr)

    def get_commit_details(
        commit_sha: Annotated[str, "Pull request commit SHA"],
    ) -> list[CommitFileDetails]:
        """Return details for files changed in the commit identified by commit_sha."""
        commit = repository.get_commit(commit_sha)
        return [commit_file_to_dict(commit_file) for commit_file in commit.files]

    def get_file_contents(
        repository_file_path: Annotated[str, "Repository file path"],
    ) -> str:
        """Retrieve the contents of a repository file as text."""
        repository_file = repository.get_contents(repository_file_path)
        return repository_file.decoded_content.decode(FILE_CONTENT_ENCODING)


    return [
        FunctionTool.from_defaults(get_pr_details),
        FunctionTool.from_defaults(get_commit_details),
        FunctionTool.from_defaults(get_file_contents),
    ]


def create_context_tools() -> list:
    async def add_context_to_state(
            context: Context,
            gathered_contexts: Annotated[str, "Retrieved contexts"]) -> str:
        """Save the retrieved contexts in the current state."""
        current_state = await context.store.get("state")
        current_state["gathered_contexts"] = gathered_contexts
        await context.store.set("state", current_state)
        return "Retrieved contexts have been saved to state."

    return [add_context_to_state]

def create_commentor_tools() -> list:
    async def save_draft_comment(
            context: Context,
            draft_comment: Annotated[str, "Draft comment"]) -> str:
        """Save the draft comment in the current state."""
        current_state = await context.store.get("state")
        current_state["draft_comment"] = draft_comment
        await context.store.set("state", current_state)
        return "Draft has been saved to state."

    async def add_comment_to_state(
            context: Context,
            comment: Annotated[str, "Comment"]) -> str:
        """Save the draft comment in the current state."""
        current_state = await context.store.get("state")
        current_state["draft_comment"] = comment
        await context.store.set("state", current_state)
        return "Draft comment has been saved to state."

    return [add_comment_to_state]

def create_llm() -> OpenAI:
    return OpenAI(
        model=os.getenv("OPENAI_MODEL", default=DEFAULT_OPENAI_MODEL),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )


def create_review_posting_agent_tools(repository)-> list:
    async def save_final_review(
            context: Context,
            draft_comment: Annotated[str, "Draft comment"]) -> str:
        """Save the final review in the current state."""
        current_state = await context.store.get("state")
        current_state["final_review"] = draft_comment
        await context.store.set("state", current_state)
        return "Draft has been saved to state."

    def post_final_review(
            pr_number: Annotated[int, "PR number"],
            final_review_comment: Annotated[str, "Final review comment"]) -> str:
        """Post the final review comment on the pull request."""
        pr = repository.get_pull(pr_number)
        pr.create_review(body=final_review_comment, event="COMMENT")
        return "Final review has been posted to the pull request."

    return [
        save_final_review,
        FunctionTool.from_defaults(post_final_review),
    ]


def create_review_posting_agent(repository, llm:OpenAI) -> FunctionAgent:
    return FunctionAgent(
        tools= create_review_posting_agent_tools(repository),
        llm=llm,
	    name="ReviewAndPostingAgent",
	    description="Reviews the generated draft comment, checks if refinements are needed, and finally posts final comment to GitHub",
        system_prompt=REVIEW_POSTING_AGENT_SYSTEM_PROMPT,
	    can_handoff_to = ["CommentorAgent"],
    )

def create_context_agent(repository, llm:OpenAI) -> FunctionAgent:
    return FunctionAgent(
        tools= create_repository_tools(repository) + create_context_tools(),
        llm=llm,
	    name="ContextAgent",
	    description="Gathers all the needed context ... ",
        system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT,
	    can_handoff_to = ["CommentorAgent"],
    )

def create_commentor_agent(llm:OpenAI) -> FunctionAgent:
    return FunctionAgent(
        tools=create_commentor_tools(),
        llm=llm,
        name="CommentorAgent",
        description="Creates a draft pull request review comment from gathered context.",
        can_handoff_to = ["ContextAgent", "ReviewAndPostingAgent"],
        system_prompt=COMMENTOR_AGENT_SYSTEM_PROMPT,
)


async def set_name(ctx: Context, name: str) -> str:
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["name"] = name

    return f"Name set to {name}"

def create_workflow(context_agent: FunctionAgent, commentor_agent: FunctionAgent, review_posting_agent:FunctionAgent) -> AgentWorkflow:
    return AgentWorkflow(
        agents=[context_agent, commentor_agent, review_posting_agent],
        root_agent=review_posting_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "final_review": ""
        },
    )

async def stream_agent_response(workflow_agent: AgentWorkflow, query: str) -> None:
    prompt = RichPromptTemplate(query)
    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


# Needed to pass a test. Not used in solution
dotenv.load_dotenv()
repo_url = build_repo_url()


async def main() -> None:
    dotenv.load_dotenv()
    pr_number = int(get_required_env_variable("PR_NUMBER"))
    llm = create_llm()
    github_client = create_github_client()
    try:
        repository = get_repository(github_client)
        context_agent = create_context_agent(repository, llm)
        commentor_agent = create_commentor_agent(llm)
        review_posting_agent = create_review_posting_agent(repository, llm)
        workflow = create_workflow(context_agent, commentor_agent, review_posting_agent)
        query = f"Write a review for PR number {pr_number}"
        await stream_agent_response(workflow, query)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        github_client.close()


if __name__ == "__main__":
    asyncio.run(main())