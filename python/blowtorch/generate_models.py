from pathlib import Path
import jinja2
import os
import json
import jsonschema
from .layers import Model


def get_template(name: str):
    template_path = Path(__file__).parent / "templates"
    loader = jinja2.FileSystemLoader(template_path)
    env = jinja2.Environment(loader=loader)
    return env.get_template(name)


def write_output(filename: str, content: str):
    with open(filename, "w+") as output_file:
        output_file.write(content)
        print(f"Successfully wrote output to {filename}")


def make_py(models: list[Model], debug: bool = False):
    template = get_template("models_template.py.jinja2")

    content = template.render(
        models=models, file=__file__, debug=debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.py")
    write_output(model_output_file, content)


def make_rs(models: list[Model], debug: bool = False):
    template = get_template("models_template.rs.jinja2")

    content = template.render(
        models=models, file=__file__, debug=debug)

    # writing out the models.rs file
    model_output_file = os.path.join("models", "models.rs")
    write_output(model_output_file, content)


def models_from_spec(spec: str, skip_validation: bool = False) -> list[Model]:
    specification_file = open(spec, "r")
    specifications = json.load(specification_file)
    if not skip_validation:
        with open(Path(__file__).parent / "schema/model-schema.schema") as schema_file:
            jsonschema.validate(specifications, json.load(schema_file))
            print("Model specification passed validation.")
    return list(map(Model, specifications))


def generate_models(spec: str, skip_validation: bool = False, debug: bool = False, ):
    os.makedirs("models", exist_ok=True)
    models = models_from_spec(spec)
    make_py(models, debug)
    make_rs(models, debug)
