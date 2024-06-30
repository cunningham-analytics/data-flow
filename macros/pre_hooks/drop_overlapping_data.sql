{% macro drop_overlapping_data(date_field) %}

    {{ log('dropping data for interval ' ~ env_var('INTERVAL_START') ~ ' to ' ~ env_var('INTERVAL_END'), info=True) }}

        delete from {{ this }}
            where {{ date_field }} between cast('{{ env_var('INTERVAL_START') }}'as date) and cast('{{ env_var('INTERVAL_END') }}' as date)

{% endmacro %}