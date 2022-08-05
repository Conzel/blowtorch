from pathlib import Path
import jinja2
import os
import json
import jsonschema
from .layers import Model


def get_template(name: str):
    """
    Looks up the template file in the folder that is located under {filePath}/templates.
    """
    template_path = Path(__file__).parent / "templates"
    loader = jinja2.FileSystemLoader(template_path)
    env = jinja2.Environment(loader=loader)
    return env.get_template(name)


def write_output(filename: str, content: str):
    """Writes the given content to the file with the given name
    and prints a success message."""
    with open(filename, "w+") as output_file:
        output_file.write(content)
        print(f"Successfully wrote output to {filename}")


def make_py(models: list[Model], debug: bool = False):
    """Renders the given models into python code."""
    template = get_template("models_template.py.jinja2")

    content = template.render(models=models, file=__file__, debug=debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.py")
    write_output(model_output_file, content)


def make_rs(models: list[Model], debug: bool = False):
    """Renders the given models into python code."""
    template = get_template("models_template.rs.jinja2")

    content = template.render(models=models, file=__file__, debug=debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.rs")
    write_output(model_output_file, content)


def models_from_spec(spec: str, skip_validation: bool = False) -> list[Model]:
    """
    Takes in the specification and returns the described models that can
    be rendered afterwards. If skip_validation is set to true, the models are
    not validated according to the jsonschema.
    """
    specification_file = open(spec, "r")
    specifications = json.load(specification_file)
    if not skip_validation:
        with open(Path(__file__).parent / "schema/model-schema.schema") as schema_file:
            jsonschema.validate(specifications, json.load(schema_file))
            print("Model specification passed validation.")
    return list(map(Model, specifications))


def generate_models(
    spec: str,
    skip_validation: bool = False,
    debug: bool = False,
):
    """
    Loads models from the given specification and turns them into
    useable python and Rust code that is written to the models folder.
    """
    os.makedirs("models", exist_ok=True)
    models = models_from_spec(spec, skip_validation=skip_validation)
    make_py(models, debug)
    make_rs(models, debug)
