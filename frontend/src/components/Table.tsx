import React, { type ReactNode } from 'react';
import classNames from 'classnames';
import './Table.css';

export interface TableColumn<T = Record<string, unknown>> {
    key: string;
    header: string;
    render?: (value: unknown, row: T, index: number) => React.ReactNode;
    sortable?: boolean;
    width?: string;
    align?: 'left' | 'center' | 'right';
}

export interface TableProps<T = Record<string, unknown>> {
    columns: TableColumn<T>[];
    data: T[];
    loading?: boolean;
    emptyMessage?: string;
    onRowClick?: (row: T, index: number) => void;
    className?: string;
    striped?: boolean;
    hoverable?: boolean;
    bordered?: boolean;
    compact?: boolean;
}

function Table<T = Record<string, unknown>>({
    columns,
    data,
    loading = false,
    emptyMessage = 'No data available',
    onRowClick,
    className,
    striped = true,
    hoverable = true,
    bordered = false,
    compact = false,
}: TableProps<T>) {
    const tableClasses = classNames(
        'table',
        {
            'table-striped': striped,
            'table-hover': hoverable,
            'table-bordered': bordered,
            'table-sm': compact,
        },
        className
    );

    const getCellValue = (row: T, column: TableColumn<T>): unknown => {
        const keys = column.key.split('.');
        let value: unknown = row;
        for (const key of keys) {
            value = (value as Record<string, unknown>)?.[key];
        }
        return value;
    };

    const renderCell = (row: T, column: TableColumn<T>, rowIndex: number): ReactNode => {
        const value = getCellValue(row, column);

        if (column.render) {
            return column.render(value, row, rowIndex);
        }

        if (value === null || value === undefined) {
            return <span className="text-muted">—</span>;
        }

        if (typeof value === 'boolean') {
            return value ? '✓' : '✗';
        }

        return <>{value}</>;
    };

    if (loading) {
        return (
            <div className="table-loading">
                <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                </div>
            </div>
        );
    }

    if (!data || data.length === 0) {
        return (
            <div className="table-empty">
                <p className="text-muted mb-0">{emptyMessage}</p>
            </div>
        );
    }

    return (
        <div className="table-responsive">
            <table className={tableClasses}>
                <thead>
                    <tr>
                        {columns.map((column) => (
                            <th
                                key={column.key}
                                style={{ width: column.width }}
                                className={classNames({
                                    'text-center': column.align === 'center',
                                    'text-end': column.align === 'right',
                                })}
                            >
                                {column.header}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {data.map((row, rowIndex) => (
                        <tr
                            key={rowIndex}
                            onClick={() => onRowClick?.(row, rowIndex)}
                            className={classNames({
                                'cursor-pointer': !!onRowClick,
                            })}
                        >
                            {columns.map((column) => (
                                <td
                                    key={`${rowIndex}-${column.key}`}
                                    className={classNames({
                                        'text-center': column.align === 'center',
                                        'text-end': column.align === 'right',
                                    })}
                                >
                                    {renderCell(row, column, rowIndex)}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default Table;
