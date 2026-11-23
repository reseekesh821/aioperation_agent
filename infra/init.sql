CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    product TEXT NOT NULL,
    status TEXT NOT NULL,
    delivered_at DATE
);

CREATE TABLE IF NOT EXISTS inventory (
    product_name TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL CHECK (quantity >= 0)
);

INSERT INTO orders (order_id, product, status, delivered_at)
VALUES
    (1001, 'MacBook Air', 'Shipped', NULL),
    (1002, 'Keyboard', 'Processing', NULL),
    (1003, 'Mouse', 'Delivered', '2026-11-01')
ON CONFLICT (order_id) DO NOTHING;

INSERT INTO inventory (product_name, quantity)
VALUES
    ('Monitor', 5),
    ('Keyboard', 12),
    ('Mouse', 0),
    ('Headphones', 8)
ON CONFLICT (product_name) DO NOTHING;