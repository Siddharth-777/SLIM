CREATE TABLE lake_readings (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    timestamp timestamptz DEFAULT now(),
    ph float,
    turbidity float,
    temperature float,
    do_level float
);
