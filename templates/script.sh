cd {{ project.config.project_dir }}

{% for operation in operations %}
{{ operation.cmd }}
{% endfor %}
