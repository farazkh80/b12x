# read_memory

Searches facts learned in previous sessions and stored with store_memory for an answer to a question.
* Facts include, but are not limited to, information about repo build and bootstrap steps, the codebase conventions, structure, logic, usage, as well as learned work-arounds and troubleshooting steps.
* Use read_memory often as it is a faster, more efficient, and more effective way to retrieve information than searching the codebase. Use it instead of grep and other search tools, when possible.
* Reference all relevant topic, specific file names, and concepts that you want to learn about in the question. For example, this question references 3 closely related topics: "What is the recommended way to bootstrap, build, and validate the project?".
* The returned answer will be in natural language and may be derived from multiple facts.

If you need information, you can use the read_memory tool to check facts you stored with store_memory in previous sessions. These facts may include:
* Conventions, preferences, or best practices that are specific to this codebase, and that might be overlooked in the future when inspecting only a limited code sample from the codebase.
* Important information about the structure or logic of the codebase.
* Commands for linting, building the code, or running tests which have been verified through a successful run.

Good questions adhere to the following criteria:
<questions_criteria>
* Are concise and specific
* Reference all relevant topics, specific file names, and concepts that you want to learn about in the question. For example, this question references 3 closely related topics: "What is the recommended way to bootstrap, build, and validate the project?".
* Use task names, tools, or language commonly used in the ecosystem.
* Ask multiple questions in a single call to read_memory, when applicable, enabling the tool to craft a more nuanced and complete response.
* Provide context on what you are trying to achieve, so that the tool can provide a more relevant answer.
</questions_criteria>

<examples>
* How can I build, test, and validate changes to src/login.ts?
* I have encountered <problem> when building the code for the first time to assess which tests pass. How can I work-around this?
* I am editing the src/admin/landingPage.tsx file. What are the style, linting, and formatting practices to follow? Has there been any user feedback I should consider?
</examples>