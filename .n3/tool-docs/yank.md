# yank

Transfer code blocks between files with smart indentation adjustment.

This tool allows you to move or copy functions, classes, methods, or arbitrary code
blocks from one file to another while automatically handling indentation differences.

WHEN TO USE YANK:
- Moving a function from one module to another
- Copying a class to a new file
- Reorganizing code within a file
- Extracting code blocks with proper indentation

WHEN NOT TO USE YANK (use Edit instead):
- Making small modifications within a file
- Renaming variables or functions
- Adding a few lines of code

WHEN TO USE YANK DELETE (instead of Edit):
- Deleting functions, classes, or large code blocks by symbol name or line range
- More token-efficient than Edit when removing code (no need to specify old_string)

SELECTION METHODS:

1. symbol - Select by function/class/method name
   Example: {"symbol": "parse_config"}
   Example: {"symbol": "MyClass.my_method"}  (for methods)

2. lines - Select by line range (1-based, inclusive)
   Example: {"lines": {"start": 10, "end": 25}}

3. pattern - Select by matching code pattern
   Example: {"pattern": "def calculate_total("}
   The pattern matches and selects the containing code block.

TARGET POSITIONING:

- after_line: Insert after a specific line number
  Example: {"after_line": 50}

- after_symbol: Insert after a named function/class
  Example: {"after_symbol": "other_function"}

- before_symbol: Insert before a named function/class
  Example: {"before_symbol": "next_function"}

- location: Position by location
  - "top": After imports/module docstring
  - "bottom": At end of file
  - "auto": Automatic (usually bottom)

INDENTATION MODES:

- "auto" (default): Detect target file's indent style and adapt
- "preserve": Keep source indentation exactly as-is
- "explicit": Use delta_levels for manual adjustment

delta_levels: Adjust indentation levels (+1 to indent, -1 to dedent)

EXAMPLES:

1. Move a function by name:
   yank(
     source_file="/src/utils.py",
     target_file="/src/helpers.py",
     selector={"symbol": "parse_config"}
   )

2. Copy lines with specific position:
   yank(
     source_file="/src/old.py",
     target_file="/src/new.py",
     selector={"lines": {"start": 50, "end": 100}},
     operation="copy",
     target_position={"after_line": 25}
   )

3. Move class with auto-indentation:
   yank(
     source_file="/src/models.py",
     target_file="/src/entities.py",
     selector={"symbol": "UserModel"},
     target_position={"after_symbol": "BaseModel"},
     indentation={"mode": "auto"}
   )

4. Preview before executing (dry run):
   yank(
     source_file="/src/a.py",
     target_file="/src/b.py",
     selector={"symbol": "important_func"},
     dry_run=True
   )

5. Delete a function (no target_file needed):
   yank(
     source_file="/src/utils.py",
     selector={"symbol": "deprecated_function"},
     operation="delete"
   )

6. Delete lines by range:
   yank(
     source_file="/src/utils.py",
     selector={"lines": {"start": 50, "end": 75}},
     operation="delete"
   )

IMPORTANT REQUIREMENTS:
- Both source and target files MUST be read first using the Read tool
- The tool validates that files haven't been modified since reading
- Operations are atomic: either both files are updated or neither is
- Use dry_run=True to preview changes before executing

ERROR HANDLING:
- If validation fails, no changes are made
- If an error occurs during execution, all changes are rolled back
- Error messages indicate which validation layer failed