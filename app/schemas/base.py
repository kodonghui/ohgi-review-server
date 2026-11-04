"""
DTO들의 부모 모델

JSON 키는 Spring, React와 동일하게 camelCase로 주고받고, 파이썬 코드에서는 snake_case 필드 이름을 사용한다.
설정을 부모 모델에 한 번만 적어 두고, DTO들이 상속받아 사용한다.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """camelCase JSON을 읽고 camelCase로 응답하는 DTO들의 부모"""

    model_config = ConfigDict(
        # alias_generator : 필드마다 camelCase 별칭을 자동으로 만든다. (customer_name -> customerName)
        alias_generator=to_camel,
        # populate_by_name : 별칭(customerName)뿐 아니라 파이썬 이름(customer_name)으로도 값을 넣을 수 있게 한다.
        populate_by_name=True,
    )
