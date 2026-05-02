# store_memory

Store a fact about the codebase in memory, so that it can be used in future code generation or review tasks. The fact should be a clear and concise statement about the codebase conventions, structure, logic, or usage. It may be based on the code itself, or on information provided by the user.

If you come across an important fact about the codebase that could help in future code review or generation tasks, beyond the current task, use the store_memory tool to store it. Facts may be gleaned from the codebase itself or from user input or feedback. Such facts might include:
* Conventions, preferences, or best practices that are specific to this codebase, and that might be overlooked in the future when inspecting only a limited code sample from the codebase.
* Important information about the structure or logic of the codebase.
* Commands for linting, building the code, or running tests which have been verified through a successful run.

<examples>
* "Use ErrKind wrapper for every public API error"
* "Prefer ExpectNoLog helper over silent nil checks in tests"
* "Always use Python typing"
* "Follow the Google JavaScript Style Guide"
* "Use html_escape as a sanitizer to avoid cross site scripting vulnerabilities"
* "The code can be built with `npm run build` and tested with `npm run test`"
</examples>

Only store facts that meet the following criteria:
<facts_criteria>
* are likely to have actionable implications to a future task
* are independent of changes you are making as part of your current task, and will remain relevant if your current code isn't merged
* are unlikely to change over time
* can't always be inferred from a limited code sample
* contain no secrets or sensitive data.
</facts_criteria>

Call store_memory once per individual fact, convention, preference, or practice. Don't forget to include the "reason" and "source" arguments in the store_memory tool call, explaining why you are storing this information and where it comes from.

Before calling store_memory, think: Will this help with future coding or code review tasks across the repository? If unsure, skip the call.