from os.path import basename

import libcst as cst
import collections
import re
import string
from dataclasses import dataclass

import black
from termcolor import colored

# Regexes to detect FIXME comments in Python (# FIXME ...) and Markdown (<!-- FIXME ... -->)
FIXME_PY_RE = re.compile(r"#\s*FIXME\b", re.IGNORECASE)
FIXME_HTML_RE = re.compile(r"<!--\s*FIXME\b", re.IGNORECASE)


def strip_trailing_spaces(s: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", s)


def preprocess_markdown(text: str) -> str:
    # Ensure there's a newline after </summary> tags in <details> sections - otherwise markdown inside doesn't render
    text = re.sub(r"</summary>(?!\n\n)\n?", "</summary>\n\n", text)

    # Add <blockquote> around <details> sections to make it clear where they end when expended:
    text = re.sub(
        r"<details>(.*?)</summary>(.*?)</details>",
        r"<details>\1</summary><blockquote>\2</blockquote></details>",
        text,
        flags=re.DOTALL,
    )

    # Add borders around hints:
    # text = re.sub(r'<details>', '<details style="border: 1px solid #ccc; padding: 5px; margin: 5px;">', text)

    text = strip_trailing_spaces(text)
    return text


class CollectImports(cst.CSTVisitor):
    """Collect the module's top-level imports, for the header of the test file.

    Only *top-level* imports are collected. An import nested inside a function or
    an `if "SOLUTION":` block belongs to that solution code, not to the test
    module: hoisting it would make the generated `*_test.py` import a package the
    section deliberately defers (e.g. the optional vLLM stack in 4.4), so every
    test in the section would fail at import time.
    """

    def __init__(self):
        super().__init__()
        self.imports = []

    def visit_Module(self, node: cst.Module):
        for stmt in node.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            for expr in stmt.body:
                if isinstance(expr, (cst.Import, cst.ImportFrom)):
                    self.imports.append(expr)


class StripSolutions(cst.CSTTransformer):
    def leave_If(self, original_node: cst.If, updated_node: cst.If):
        """Strip out contents of if 'SOLUTION': and if 'SKIP': block."""
        if isinstance(updated_node.test, cst.SimpleString):
            # Remove quotes to get the actual string value
            test_value = updated_node.test.value.strip("'\"")
            if test_value == "SOLUTION":
                if updated_node.orelse:
                    # Extract the body from the else clause instead of returning the entire Else node
                    if isinstance(updated_node.orelse, cst.Else):
                        else_block = updated_node.orelse.body
                        else_body = list(else_block.body)
                        # libcst stores bare comments that trail the last statement in
                        # a block on IndentedBlock.footer; without this they'd be
                        # silently dropped when we flatten the else body, so
                        # re-attach them as leading comments on a synthetic `pass`.
                        footer_comments = [
                            line for line in else_block.footer if line.comment is not None
                        ]
                        if footer_comments:
                            else_body.append(
                                cst.SimpleStatementLine(
                                    body=[cst.Pass()],
                                    leading_lines=footer_comments,
                                )
                            )
                        if else_body:
                            return cst.FlattenSentinel(else_body)
                        else:
                            return cst.SimpleStatementLine(
                                body=[cst.Expr(value=cst.SimpleString('"TODO: YOUR CODE HERE"'))]
                            )
                    return updated_node.orelse
                return cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString('"TODO: YOUR CODE HERE"'))])
            if test_value == "SKIP":
                return cst.RemovalSentinel.REMOVE
            if test_value == "REFERENCE_ONLY":
                return cst.RemovalSentinel.REMOVE
            if test_value == "TEST_FIXTURE":
                # Unwrap to top level. If there's an else branch, the else body
                # is what participants see in the instructions; otherwise the
                # if body is shown.
                if updated_node.orelse and isinstance(updated_node.orelse, cst.Else):
                    else_body = list(updated_node.orelse.body.body)
                    if else_body:
                        return cst.FlattenSentinel(else_body)
                return cst.FlattenSentinel(updated_node.body.body)
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Strip out remainder of function body after 'SOLUTION' constant."""
        new_body: list = []
        for stmt in updated_node.body.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for expr in stmt.body:
                    if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
                        # Remove quotes to get the actual string value
                        value = expr.value.value.strip("'\"")
                        if value == "SOLUTION":
                            new_body.append(cst.SimpleStatementLine(body=[cst.Pass()]))
                            return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))
            new_body.append(stmt)
        return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))


class ExtractSolutionBlocks(cst.CSTTransformer):
    """Extract only SOLUTION blocks and remove test functions"""

    def leave_If(self, original_node: cst.If, updated_node: cst.If):
        """Keep only contents of if 'SOLUTION': blocks."""
        if isinstance(updated_node.test, cst.SimpleString):
            test_value = updated_node.test.value.strip("'\"")
            if test_value == "SOLUTION":
                # Return only the body of the SOLUTION block
                return cst.FlattenSentinel(updated_node.body.body)
            if test_value == "SKIP":
                return cst.RemovalSentinel.REMOVE
            if test_value == "REFERENCE_ONLY":
                return cst.FlattenSentinel(updated_node.body.body)
            if test_value == "TEST_FIXTURE":
                # Reference and test files always use the if body.
                return cst.FlattenSentinel(updated_node.body.body)
        return updated_node


class CollectTestFixtures(cst.CSTVisitor):
    """Collect the if-body of top-level if "TEST_FIXTURE": blocks."""

    def __init__(self):
        super().__init__()
        self.fixtures: list[cst.BaseStatement] = []

    def visit_Module(self, node: cst.Module):
        for stmt in node.body:
            if (
                isinstance(stmt, cst.If)
                and isinstance(stmt.test, cst.SimpleString)
                and stmt.test.value.strip("'\"") == "TEST_FIXTURE"
            ):
                self.fixtures.extend(stmt.body.body)


class StripTestFunctions(cst.CSTTransformer):
    """Remove all functions that start with test_"""

    def __init__(self, test_file_name):
        super().__init__()
        self.test_functions = []
        self.module_name = test_file_name.replace(".py", "").replace("/", ".")

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef):
        if updated_node.name.value.startswith("test_"):
            self.test_functions.append(updated_node)
            module_parts = self.module_name.split(".")

            # Construct the module for the import statement
            module = cst.Name(value=module_parts[-1])

            import_node = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=module,
                        names=[cst.ImportAlias(name=cst.Name(value=updated_node.name.value))],
                    )
                ]
            )

            return import_node
        return updated_node


@dataclass
class Snippet:
    language: str
    text: str


def is_toplevel_string_constant(node):
    """Return True if this is a string constant that's unindented."""
    if not isinstance(node, cst.SimpleStatementLine):
        return False

    if len(node.body) != 1 or not isinstance(node.body[0], cst.Expr):
        return False

    expr = node.body[0]
    if not isinstance(expr.value, cst.BaseString):
        return False

    # Check if it's at toplevel by examining indentation
    # In libcst, toplevel statements have no leading whitespace
    return not hasattr(node, "indent") or node.indent == ""


TOC_LEVELS = [
    "-",
    "    -",
    "        -",
]

TOC_RE = re.compile(r"^(#+) (.+)$")
TOC_MARKER = "<!-- toc -->"
SLUG_REMOVE_CHARS_REGEX = re.compile(r"[!\"#$%&'()*+,./:;<=>?@\[\\\]^`{|}~]+")


@dataclass
class TOCEntry:
    title: str
    level: int  # zero-indexed, e.g. # is level 0
    slug: str  # spaces to dashes, possible -dash-number


# This is very incomplete but works well enough for now!
TAG_RE = re.compile(r"<(/?)(\w+)(?: .*?)?(/?)>")
NO_CLOSE_TAGS = ["br", "img"]


def check_html_tags(text: str) -> list[str]:
    """Return a list of warnings about mismatched HTML tags."""
    tags = TAG_RE.findall(text)
    tagname_stack: list[str] = []
    warnings: list[str] = []
    for tag in tags:
        close_slash, tagname, end_slash = tag
        if close_slash:
            if end_slash:
                warnings.append(f"WARNING: malformed HTML tag {tag} ")
            else:
                if tagname_stack:
                    top = tagname_stack.pop()
                    if top == tagname:
                        continue  # proper close
                    else:
                        warnings.append(f"WARNING: should have closed {top} but found {tag}")
                else:
                    warnings.append(f"WARNING: tried to close {tag} but no tags open")
        else:
            if end_slash:
                continue  # is self-closing
            elif tagname in NO_CLOSE_TAGS:
                continue
            else:
                tagname_stack.append(tagname)
    if tagname_stack:
        warnings.append(f"WARNING: tags not closed: {tagname_stack}")
    return warnings


class InstructionMaker(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.snippets = []
        self.mode = black.Mode(line_length=120)  # type: ignore
        self.toc_entries: list[TOCEntry] = []
        self.counters = collections.defaultdict(int)
        self.module = None

    def visit_Module(self, node: cst.Module):
        self.module = node
        for stmt in node.body:
            if is_toplevel_string_constant(stmt):
                # Extract the string value
                string_node = stmt.body[0].value  # type: ignore
                text = string_node.raw_value
                text = preprocess_markdown(text)
                text = self._maybe_add_toc(text)
                warnings = check_html_tags(text)
                if warnings:
                    print("Bad HTML tags in statement")
                    print("\n".join(warnings))
                self.snippets.append(Snippet("markdown", text))

            else:
                # Convert back to source code
                src = self.module.code_for_node(stmt)
                if self.snippets and self.snippets[-1].language == "python":
                    self.snippets[-1].text += src
                else:
                    self.snippets.append(Snippet("python", src))

    def dump(self, fp, python_prefix_snippet):
        texts = []
        last_language = None
        first_python_snippet = True
        for snippet in self.snippets:
            if snippet.language == "markdown":
                text = snippet.text
                index = text.find(TOC_MARKER)
                if index != -1:
                    toc = self._dump_toc()
                    text = text[:index] + toc + text[index + len(TOC_MARKER) :]
                texts.append(text)
            elif snippet.language == "python":
                # pretty = black.format_str(snippet.text, mode=self.mode) // TODO: enable or keep disabled?
                # Need two newlines before code block
                extra_newline = "\n" if last_language == "markdown" else ""
                prefix = python_prefix_snippet + "\n" if first_python_snippet else ""
                text = re.sub(r"\n+$", "", snippet.text)  # remove trailing newlines
                texts.append(f"{extra_newline}```python\n{prefix}{text}\n```")
                first_python_snippet = False
            last_language = snippet.language
        fp.write("\n".join(texts))

    def _maybe_add_toc(self, text: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            m = TOC_RE.match(line)
            if m is not None:
                pounds, header_text = m.groups()
                prefix = header_text.replace(" ", "-")
                count = self.counters[prefix]
                slug = f"{prefix}-{count}" if count > 0 else prefix
                slug = slug.lower()  # VSCode only wants lowercase slugs
                slug = SLUG_REMOVE_CHARS_REGEX.sub("", slug)
                level = len(pounds) - 2
                if level < 0:
                    continue  # Don't need to repeat toplevel
                if level >= len(TOC_LEVELS):
                    raise ValueError(f"TOC doesn't yet support header level {level}: {header_text}")
                entry = TOCEntry(title=header_text, level=level, slug=slug)
                self.toc_entries.append(entry)
                self.counters[prefix] = count + 1
                # lines[i] = f'{pounds} <a id="#{slug}">{header_text}</a>'
        return "\n".join(lines) + "\n"  # trailing newline is needed for MD031

    def _dump_toc(self) -> str:
        lines = ["## Table of Contents", ""]
        for e in self.toc_entries:
            line = f"{TOC_LEVELS[e.level]} [{e.title}](#{e.slug})"
            lines.append(line)
        return "\n".join(lines)


def warn_fixme(text: str, file_name: str) -> None:
    """Emit a warning for each line that contains a FIXME comment."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        if FIXME_PY_RE.search(line) or FIXME_HTML_RE.search(line):
            print(
                colored(
                    f"WARNING: FIXME comment found in {file_name}:{lineno}: {line.strip()}",
                    "yellow",
                )
            )


def build(input_fd, output_instructions_fd, output_test_fd):
    print(f"Building: {input_fd.name} -> {output_instructions_fd.name}, {output_test_fd.name}")
    input_str = input_fd.read()

    # Warn about any FIXME comments present in the original source
    warn_fixme(input_str, input_fd.name)

    module = cst.parse_module(input_str)

    # Collect imports from the original file
    import_collector = CollectImports()
    module.visit(import_collector)

    # Collect test fixtures (if-body of if "TEST_FIXTURE": blocks)
    fixture_collector = CollectTestFixtures()
    module.visit(fixture_collector)

    # Strip solutions for template
    without_solutions = module.visit(StripSolutions())

    # Move test functions from the solutions file to the test file
    test_extractor = StripTestFunctions(test_file_name=output_test_fd.name)
    without_tests = without_solutions.visit(test_extractor)
    if test_extractor.test_functions:
        # Write imports first
        output_test_fd.write(
            "\n".join(
                [
                    "# Allow imports from parent directory",
                    "import sys",
                    "import os",
                    "sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))",
                    "",
                    "",
                ]
            )
        )
        import_output = [module.code_for_node(import_node) for import_node in import_collector.imports]
        output_test_fd.write("\n".join(import_output) + "\n\n")
        # Write test fixtures so tests can reference shared setup data
        if fixture_collector.fixtures:
            fixture_output = [module.code_for_node(stmt) for stmt in fixture_collector.fixtures]
            output_test_fd.write("\n".join(fixture_output) + "\n\n")
        # Then write test functions
        test_output = [module.code_for_node(test_func) for test_func in test_extractor.test_functions]
        output_test_fd.write("\n\n".join(test_output))

    # Generate the instructions markdown from the solutions (without tests)
    sm = InstructionMaker()
    without_tests.visit(sm)
    sm.dump(output_instructions_fd, "")


def build_reference_py(input_fd, output_reference_fd, tests_file_path: str):
    print(f"Building: {input_fd.name} -> {output_reference_fd.name}")
    input_str = input_fd.read()
    module = cst.parse_module(input_str)

    # Extract only SOLUTION blocks and remove test functions
    solution_extractor = ExtractSolutionBlocks()
    reference_code = module.visit(solution_extractor)

    test_extractor = StripTestFunctions(test_file_name=tests_file_path)
    without_tests = reference_code.visit(test_extractor)

    # Write the reference code
    output_reference_fd.write(without_tests.code)
