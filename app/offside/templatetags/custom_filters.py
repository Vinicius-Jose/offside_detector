from django import template

register = template.Library()


@register.filter
def subtract(value_1: float, value_2: float) -> float:
    return value_1 - value_2


@register.filter
def positive(value: float) -> float:
    return abs(value)


@register.filter
def multiply(value_1: float, value_2: float) -> float:
    return value_1 * value_2
