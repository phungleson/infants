require 'csv'
require 'json'
require 'finishing_moves'

scaled_values = {}

CSV.foreach('infants_columns.csv') do |row|
  column_name, unique_values = row
  unique_values = JSON.parse(unique_values)

  values = unique_values.select { |v| v != nil }

  if values.size > 0 && values[0].numeric?
    unique_values = unique_values.map(&:to_f)
    min = unique_values.min
    max = unique_values.max

    scaled_values[column_name] = {
      min: min,
      max: max,
    }
  end
end

csv = CSV.open('infants_columns_stats.csv', 'wb')

scaled_values.each do |k, values|
  csv << [k, values.to_json]
end